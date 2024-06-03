from enum import Enum
from pydantic import BaseModel, Field
from fnmatch import fnmatch
from typing import Optional, List, Union, Dict
import base64
import logging
from langcraft.action import *
from langcraft.action import ActionBrief, ActionResult
from langcraft.llm.llm_models import LLMs

"""
Module for defining the base classes for language actions.
"""


#################################################
class MessageRole(Enum):
    """
    Enum for the role of the party uttering a message in a chat.
    """

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_REQUEST = "tool_request"
    TOOL_RESULT = "tool_result"


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
class Message(BaseModel):
    """
    Represents a message with text and images.
    """

    text: str = Field(description="The text part of the message.")
    images: Optional[List[Image]] = Field(
        description="The images in the message.",
        default=[],
    )


#################################################
class ToolRequest(BaseModel):
    """
    Represents a tool execution request.
    """

    request_id: str = Field(description="The ID of the request.")
    tool_name: str = Field(description="The name of the tool.")
    tool_input: Dict = Field(description="The input to the tool.")


#################################################
class ToolResponse(BaseModel):
    """
    Represents a tool execution result.
    """

    request_id: str = Field(description="The ID of the request.")
    tool_name: str = Field(description="The name of the tool.")
    tool_result: str = Field(description="The result of the tool execution.")


#################################################
class ConversationTurn(BaseModel):
    """
    Represents a single turn in a conversation.
    """

    role: MessageRole = Field(description="The role of party uttering the message.")
    message: Union[Message, ToolRequest, ToolResponse] = Field(
        description="The content of the respective party's message."
    )


#################################################
class LanguageActionBrief(ActionBrief):
    """
    Base class for language action briefs.
    """

    model_name: Optional[str] = Field(
        description="The name of the LLM to use.", default=None
    )

    system: Optional[str] = Field(description="The system prompt.", default=None)

    tools: List[str] = Field(
        description="The names of actions accessible as tools.", default=[]
    )

    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate.", default=None
    )

    temperature: Optional[float] = Field(
        description="The sampling temperature.", default=0.0
    )

    stop: Optional[str] = Field(description="The stop sequence.", default=None)

    class Config:
        protected_namespaces = ()


#################################################
class CompletionBrief(LanguageActionBrief):
    """
    A brief for a completion language action.
    """

    prompt: Union[str, Message] = Field(
        description="The prompt, either plain text or a structured message."
    )


#################################################
class ChatBrief(LanguageActionBrief):
    """
    A brief for a chat language action.
    """

    conversation: List[ConversationTurn] = Field(
        description="The conversation history."
    )


#################################################
class CompletionResult(ActionResult):
    """
    Represents the result of a completion action.
    """

    response: str = Field(description="The completion.")

    model_name: str = Field(description="The name of the LLM used for completion.")
    input_tokens: Optional[int] = Field(
        description="The number of tokens in the input prompt.", default=None
    )
    output_tokens: Optional[int] = Field(
        description="The number of tokens in the output completion.", default=None
    )

    class Config:
        protected_namespaces = ()


#################################################
class ChatResult(CompletionResult):
    """
    Represents the result of a chat action.
    """

    conversation: List[ConversationTurn] = Field(
        description="The updated conversation history."
    )


#################################################
class LanguageAction(Action):
    """
    Base for LLM actions.
    """

    def _preprocess(self, briefs: List[ActionBrief]):
        """
        Preprocesses the given list of action briefs to resolve the model name if needed.

        Args:
            briefs (List[ActionBrief]): The list of action briefs to be preprocessed.

        Raises:
            ValueError: If an invalid brief type is encountered.
        """
        # get the model and max_tokens settings
        for brief in briefs:
            if isinstance(brief, LanguageActionBrief):
                brief.model_name = LLMs.get_model_name(brief.model_name)

                if LLMs.get_model(brief.model_name) is None:
                    raise ValueError(f"Model not found: {brief.model_name}")

                if not brief.max_tokens:
                    brief.max_tokens = LLMs.get_max_output_tokens(brief.model_name)
            else:
                raise ValueError(
                    f"Invalid brief type: {type(brief)}; expected LanguageActionBrief."
                )

            # log request
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
            input_token_cost = LLMs.get_prompt_cost(
                result.model_name, result.input_tokens
            )
            output_token_cost = LLMs.get_completion_cost(
                result.model_name, result.output_tokens
            )
            result.cost = input_token_cost + output_token_cost

            # log requresultest
            logging.getLogger("langcraft").info(
                result.__class__.__name__ + ":\n" + result.model_dump_json(indent=2)
            )

    def _compile_tools(self, tool_names: List[str]) -> List[Dict]:
        """
        Compile a list of tools into a list of dictionaries.

        Args:
            tools (List[str]): A list of names of tools.

        Returns:
            List[Dict]: A list of dictionaries containing the compiled tool information.

        """
        return [
            {
                "name": action_descriptor.name,
                "description": action_descriptor.description,
                "input_schema": action_descriptor.brief.to_schema(),
            }
            for action_descriptor in list(
                map(lambda tool_name: Actions.get(tool_name), tool_names)
            )
        ]


#################################################
class ChatAction(LanguageAction):
    """
    An action that delegates to an LLM to complete a prompt.
    """

    @classmethod
    def get_descriptor(cls) -> ActionDescriptor:
        return ActionDescriptor(
            name="llm-chat",
            description="Generates a completion for a chat conversation, using an LLM.",
            brief=ChatBrief,
            action=cls,
            result=ChatResult,
        )

    _chat_action_implementations: Dict[str, LanguageAction] = {}

    @classmethod
    def register_implementation(
        cls, model_name_matchers: List[str], implementation: LanguageAction
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
                    f"Model matcher for ChatAction already registered: {model_name_matcher}"
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

    def run(self, brief: ActionBrief) -> ActionResult:
        """
        Executes the action specified by the given ActionBrief.

        Args:
            brief (ActionBrief): The ActionBrief object containing the necessary information for the action.

        Returns:
            ActionResult: The result of executing the action.
        """
        model_name = brief.model_name
        implementation = ChatAction._get_implementation(model_name)
        return implementation().run(brief)

    def run_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs a batch of action briefs.

        Args:
            briefs (List[ActionBrief]): A list of action briefs.

        Returns:
            List[ActionResult]: A list of action results.
        """
        model_name = briefs[0].model_name
        implementation = ChatAction._get_implementation(model_name)
        return implementation().run_batch(briefs)


#################################################
Actions.register(ChatAction.get_descriptor())


#################################################
class CompletionAction(LanguageAction):
    """
    An action that uses an LLM to complete a prompt.
    """

    @classmethod
    def get_descriptor(cls) -> ActionDescriptor:
        return ActionDescriptor(
            name="llm-complete",
            description="Generates a completion for a prompt, using an LLM.",
            brief=CompletionBrief,
            action=cls,
            result=CompletionResult,
        )

    def run(self, brief: ActionBrief) -> ActionResult:
        """
        Runs the action with the given ActionBrief, delegating to ChatAction.

        Args:
            brief (ActionBrief): The ActionBrief containing the necessary information for the action.

        Returns:
            ActionResult: The result of running the action.
        """
        return ChatAction().run(
            ChatBrief(
                model_name=brief.model_name,
                tools=brief.tools,
                conversation=[
                    ConversationTurn(
                        role=MessageRole.USER,
                        message=(
                            brief.prompt
                            if isinstance(brief.prompt, Message)
                            else Message(text=brief.prompt)
                        ),
                    )
                ],
            )
        )

    def run_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Runs the action with the given ActionBrief, delegating to ChatAction.

        Args:
            brief (ActionBrief): The ActionBrief containing the necessary information for the action.

        Returns:
            ActionResult: The result of running the action.
        """
        return ChatAction().run(
            [
                ChatBrief(
                    model_name=brief.model_name,
                    conversation=[
                        ConversationTurn(
                            role=MessageRole.USER,
                            message=(
                                brief.prompt
                                if isinstance(brief.prompt, Message)
                                else Message(text=brief.prompt)
                            ),
                        )
                    ],
                )
                for brief in briefs
            ]
        )


#################################################
Actions.register(CompletionAction.get_descriptor())
