from enum import Enum
from pydantic import BaseModel, Field
from fnmatch import fnmatch
from typing import Optional, List, Union, Dict
import base64
import json
import logging
import click
from langcraft.action import *
from langcraft.action import ActionBrief, ActionResult
from langcraft.utils import extract_tag
from langcraft.llm.llm_models import LLMs

"""
Module for defining the classes for completion actions.
"""


#################################################
class Image(BaseModel):
    """
    Represents an image in a message.
    """

    mime_type: str = Field(description="The MIME type of the image.")
    image_data: str = Field(description="The base64-encoded image data.", exclude=True)

    @classmethod
    def from_file(cls, filename: str) -> str:
        """
        Load an image from a file and return it as a base64-encoded string.

        Args:
            filename (str): The path to the image file.

        Returns:
            str: The base64-encoded string representation of the image.

        Raises:
            ValueError: If the image format is not supported.
        """
        # determine MIME type
        if filename.endswith(".png"):
            mime_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image format: {filename}")

        with open(filename, "rb") as f:
            data = f.read()
            return Image(
                mime_type=mime_type, image_data=base64.b64encode(data).decode("utf-8")
            )


#################################################
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


#################################################
class Message(BaseModel):
    """
    Represents a message with text and images.
    """

    text: str = Field(description="The text part of the message.")

    images: Optional[List[Image]] = Field(
        description="The images in the message.",
        default=None,
    )


#################################################
class ToolCallRequest(BaseModel):
    """
    Represents a tool call request.
    """

    request_id: str = Field(description="The ID of the call request.")

    tool_name: str = Field(description="The name of the tool.")

    # actually an ActionBrief, but Pydantic won't have it
    tool_arguments: object = Field(description="The input arguments for the tool.")

    def run_tool(self) -> ActionResult:
        """
        Executes the specified tool with the given arguments.

        Returns:
            An instance of ActionResult representing the result of the tool execution.
        """
        return Actions.get(self.tool_name).action().run(self.tool_arguments)


#################################################
class ToolCallResult(BaseModel):
    """
    Represents a tool call result.
    """

    request_id: str = Field(description="The ID of the request.")

    tool_name: str = Field(description="The name of the tool.")

    # actually an ActionResult, but Pydantic won't have it
    tool_result: object = Field(description="The result of the tool execution.")


#################################################
class ConversationTurn(BaseModel):
    """
    Represents a single turn in a conversation.
    """

    role: str = Field(description="The role uttering the message.")

    message: Optional[Message] = Field(
        description="The content of the respective party's message.", default=None
    )

    def _text(self):
        """
        Returns the text part of the message.
        """
        if self.message is None:
            return ""

        return self.message.text or ""

    def to_str(self):
        """
        Returns the message for human display.
        """
        return f"{self.role.upper()}\n\n{self._text()}\n"

        return m

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Creates a ConversationTurn from a dictionary.

        Args:
            data (Dict): The dictionary to create the ConversationTurn from.

        Returns:
            ConversationTurn: The generated ConversationTurn.
        """
        if USER_ROLE == data["role"]:
            return UserConversationTurn(message=Message(**data["message"]))
        elif ASSISTANT_ROLE == data["role"]:
            return AssistantConversationTurn(message=Message(**data["message"]))

        raise ValueError("Invalid ConversationTurn data.")


#################################################
class UserConversationTurn(ConversationTurn):
    """
    Represents a single turn in a conversation by a user.
    """

    role: str = Field(
        description="The role uttering the message.",
        default=USER_ROLE,
        frozen=True,
    )

    tool_call_results: Optional[List[ToolCallResult]] = Field(
        description="The tool responses, if any.", default=None
    )

    def to_str(self):
        """
        Returns the message for human display.
        """
        tool_str = ""
        for tool_result in self.tool_call_results or []:
            tool_str += f"---\n{tool_result.tool_name}:\n{tool_result.tool_result.model_dump_json(indent=2)}\n"

        return super().to_str() + tool_str

    @classmethod
    def from_text(cls, text: str, **kwargs):
        """
        Creates a UserConversationTurn from a text.

        Args:
            text (str): The prompt for the conversation.

        Returns:
            UserConversationTurn: The generated UserConversationTurn.
        """
        return cls(
            message=Message(text=text),
            **kwargs,
        )


#################################################
class AssistantConversationTurn(ConversationTurn):
    """
    Represents a single turn in a conversation by the assistant.
    """

    role: str = Field(
        description="The role uttering the message.",
        default=ASSISTANT_ROLE,
        frozen=True,
    )

    tool_call_requests: Optional[List[ToolCallRequest]] = Field(
        description="The tool call requests, if any.",
        default=None,
    )

    def to_str(self):
        """
        Returns the message for human display.
        """
        tool_str = ""
        for tool_request in self.tool_call_requests or []:
            tool_str += f"---\n{tool_request.tool_name}:\n{tool_request.tool_arguments.model_dump_json(indent=2)}\n"

        return super().to_str() + tool_str

    def is_requesting_tool_calls(self):
        """
        Checks if there are any tool call requests in the conversation turn.

        Returns:
            bool: True if there are pending tool calls, False otherwise.
        """
        return self.tool_call_requests and len(self.tool_call_requests) > 0

    def run_tools(self) -> List[ToolCallResult]:
        """
        Runs the tools requested by iterating over the tool_requests list and calling the run_tool method for each tool_request.
        """
        click.echo(
            f"Running tools {','.join([request.tool_name for request in self.tool_call_requests])}"
        )
        return [
            ToolCallResult(
                request_id=tool_call_request.request_id,
                tool_name=tool_call_request.tool_name,
                tool_result=tool_call_request.run_tool(),
            )
            for tool_call_request in self.tool_call_requests or []
        ]


#################################################
class CompletionBrief(ActionBrief):
    """
    Class for a completion brief.
    """

    model_name: Optional[str] = Field(
        description="The name of the LLM to use.", default=None
    )

    system: Optional[str] = Field(description="The system prompt.", default=None)

    conversation: List = Field(description="The conversation so far.")

    tools: List[str] = Field(
        description="The names of actions accessible as tools.", default=[]
    )

    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.", default=None
    )

    temperature: Optional[float] = Field(
        description="The sampling temperature.", default=0.7
    )

    stop: Optional[str] = Field(description="The stop sequence.", default=None)

    class Config:
        # to prevent complaints about the model_name field
        protected_namespaces = ()

    def has_pending_tool_calls(self):
        """
        Checks if there are any pending tool calls in the last conversation turn.

        Returns:
            bool: True if there are pending tool calls, False otherwise.
        """
        return (
            len(self.conversation) > 0
            and self.conversation[-1].role == ASSISTANT_ROLE
            and self.conversation[-1].is_requesting_tool_calls()
        )

    def extend_conversation(
        self, turn: ConversationTurn, run_tools: bool = True
    ) -> bool:
        """
        Extends the conversation by appending a new turn and optionally running any pending tool calls.

        Args:
            turn (ConversationTurn): The turn to be added to the conversation.
            run_tools (bool, optional): Whether to run any pending tool calls. Defaults to True.

        Returns:
            bool: True if tools were called.
        """
        self.conversation.append(turn)

        if self.has_pending_tool_calls() and run_tools:
            self.extend_conversation(
                UserConversationTurn.from_text(
                    text="", tool_call_results=self.conversation[-1].run_tools()
                )
            )

            return True

        return False

    @classmethod
    def from_prompt(cls, prompt: Union[str, Message], **kwargs):
        """
        Creates a ConversationBrief from a prompt.

        Args:
            prompt (Union[str,Message]): The prompt for the conversation.

        Returns:
            ConversationBrief: The generated ConversationBrief.
        """
        return cls(
            conversation=[
                UserConversationTurn(
                    message=(
                        prompt if isinstance(prompt, Message) else Message(text=prompt)
                    )
                )
            ],
            **kwargs,
        )


#################################################
class CompletionResult(ActionResult):
    """
    Represents the result of a completion action.
    """

    model_name: str = Field(
        description="The name of the LLM model used for completion."
    )

    input_tokens: Optional[int] = Field(
        description="The number of tokens in the input prompt.", default=None
    )

    output_tokens: Optional[int] = Field(
        description="The number of tokens in the output completion.", default=None
    )

    conversation_turn: object = Field(
        description="The response turn in the conversation."
    )

    class Config:
        # to prevent complaints about the model_name field
        protected_namespaces = ()

    def get_text_completion(self) -> str:
        """
        Gets the text completion from the conversation turn.

        Returns:
            str: The text completion.
        """
        return self.conversation_turn.message.text or ""

    def extract_tag(self, tag) -> List[str]:
        """
        Extracts all string contents enclosed in any set of pairs of XML tags in the given string.

        Parameters:
        tag (str): The XML tag to search for.

        Returns:
        List[str]: The extracted string contents enclosed in the XML tags.
        """
        return extract_tag(tag, self.get_text_completion())


#################################################
class CompletionAction(Action):
    """
    An action that delegates to an LLM to complete a conversation.
    """

    @classmethod
    def get_descriptor(cls) -> ActionDescriptor:
        return ActionDescriptor(
            name="llm-completion",
            description="Generates a completion for a conversation, using an LLM.",
            brief=CompletionBrief,
            action=cls,
            result=CompletionResult,
        )

    # dictionary of model name matchers to implementations
    _chat_action_implementations: Dict[str, Action] = {}

    @classmethod
    def register_implementation(
        cls, model_name_matchers: List[str], implementation: Action
    ):
        """
        Register an implementation for a given model name matcher.

        Args:
            cls (type): The class to register the implementation for.
            model_name_matchers (List[str]): A list of model name matchers to register.
            implementation (LanguageAction): The implementation to register.

        Raises:
            ValueError: If a model name matcher is already registered for the given class.

        """
        for model_name_matcher in model_name_matchers:
            if model_name_matcher in cls._chat_action_implementations:
                raise ValueError(
                    f"Model matcher already registered: {model_name_matcher}"
                )

            cls._chat_action_implementations[model_name_matcher] = implementation

    @classmethod
    def _get_implementation(cls, model_name: str):
        """
        Get the implementation for the given model name.

        Args:
            model_name (str): The name of the model.

        Returns:
            implementation: The implementation associated with the model name.

        Raises:
            ValueError: If the model name is not found.
        """
        for (
            model_name_matcher,
            implementation,
        ) in cls._chat_action_implementations.items():
            if fnmatch(model_name, model_name_matcher):
                return implementation

        raise ValueError(f"Model not found: {model_name}")

    @classmethod
    def run(cls, brief: ActionBrief) -> ActionResult:
        """
        Executes the action specified by the given ActionBrief.

        Args:
            brief (ActionBrief): The ActionBrief object containing the necessary information for the action.

        Returns:
            ActionResult: The result of executing the action.
        """

        # get the model and max_tokens settings
        model_name = LLMs.resolve_model(brief.model_name)
        if model_name is None:
            raise ValueError(f"Model not found: {brief.model_name}")
        brief.model_name = model_name

        if not brief.max_tokens:
            brief.max_tokens = LLMs.get_max_output_tokens(brief.model_name)

        implementation = CompletionAction._get_implementation(brief.model_name)

        return implementation().run(brief)

    @classmethod
    def run_with_tools(
        cls, brief: CompletionBrief, max_iterations=10
    ) -> CompletionResult:
        """
        Run the language model with additional tools.

        Args:
            brief (CompletionBrief): The completion brief to be processed.
            max_iterations (int, optional): The maximum number of iterations to extend the conversation. Defaults to 10.

        Returns:
            CompletionResult: The result of the completion process.
        """

        result = CompletionAction.run(brief)

        if brief.extend_conversation(result.conversation_turn, max_iterations > 0):
            return CompletionAction.run_with_tools(brief, max_iterations - 1)

        return result

    @classmethod
    def run_single_completion(
        cls, model_name: str, prompt: str, system: str = "", tag_to_extract: str = None
    ) -> str:
        """
        Run a single completion for a given model and prompt.

        Args:
            model_name (str): The name of the model to use.
            prompt (str): The prompt to use.
            system (str, optional): The system prompt. Defaults to "".

        Returns:
            str: The completion result.
        """
        result = cls.run(
            CompletionBrief.from_prompt(
                model_name=model_name,
                prompt=prompt,
                system=system,
            )
        )

        if tag_to_extract:
            return result.extract_tag(tag_to_extract)
        else:
            return result.get_text_completion()

    @classmethod
    def run_batch(cls, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of action briefs.

        Args:
            briefs (List[ActionBrief]): A list of action briefs.

        Returns:
            List[ActionResult]: A list of action results.
        """
        # get the model and max_tokens settings
        model_name = None
        for brief in briefs:
            model_name = LLMs.resolve_model(brief.model_name)

            if model_name is None:
                raise ValueError(f"Model not found: {brief.model_name}")
            if model_name != brief.model_name:
                raise ValueError(
                    f"Model mismatch in batch: {model_name} != {brief.model_name}"
                )
            brief.model_name = model_name

            if not brief.max_tokens:
                brief.max_tokens = LLMs.get_max_output_tokens(brief.model_name)

        implementation = CompletionAction._get_implementation(model_name)

        return implementation().run_batch(briefs)


#################################################
Actions.register(CompletionAction.get_descriptor())


#################################################
class CompletionDelegateAction(Action):
    """
    Base class for actual completion implementations.
    """

    def _preprocess(self, briefs: List[ActionBrief]):
        """
        Preprocesses the given list of ActionBrief objects.

        Args:
            briefs (List[ActionBrief]): A list of ActionBrief objects to be preprocessed.
        """
        # log request
        for brief in briefs:
            logging.getLogger("langcraft").info(
                brief.__class__.__name__ + ":\n" + brief.model_dump_json(indent=2)
            )

        super()._preprocess(briefs)

    def _postprocess(self, results: List[ActionResult]):
        """
        Postprocesses the results by calculating the cost for each result.

        Args:
            results (List[ActionResult]): The list of results to be postprocessed.
        """
        # calculate the cost & log
        for result in results:
            if isinstance(result, CompletionResult):
                input_token_cost = LLMs.get_prompt_cost(
                    result.model_name, result.input_tokens
                )
                output_token_cost = LLMs.get_completion_cost(
                    result.model_name, result.output_tokens
                )
                result.cost = input_token_cost + output_token_cost

            # log result
            logging.getLogger("langcraft").info(
                result.__class__.__name__ + ":\n" + result.model_dump_json(indent=2)
            )

        super()._postprocess(results)
