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
        default=None,
    )


#################################################
class ToolRequest(BaseModel):
    """
    Represents a tool execution request.
    """

    request_id: str = Field(description="The ID of the request.")
    tool_name: str = Field(description="The name of the tool.")
    tool_input: object = Field(description="The brief for the tool.")

    def run_tool(self) -> ActionResult:
        return Actions.get(self.tool_name).action().run(self.tool_input)


#################################################
class ToolResponse(BaseModel):
    """
    Represents a tool execution result.
    """

    request_id: str = Field(description="The ID of the request.")
    tool_name: str = Field(description="The name of the tool.")
    tool_result: object = Field(description="The result of the tool execution.")


#################################################
class ConversationTurn(BaseModel):
    """
    Represents a single turn in a conversation.
    """

    role: MessageRole = Field(description="The role of the message.")
    message: Optional[Message] = Field(
        description="The content of the respective party's message.", default=None
    )


#################################################
class UserConversationTurn(ConversationTurn):
    """
    Represents a single turn in a conversation by a user.
    """

    role: MessageRole = Field(
        description="The role of the message.", default=MessageRole.USER, frozen=True
    )
    tool_results: List[ToolResponse] = Field(
        description="The tool responses; only if this is a user turn.", default=None
    )


#################################################
class AssistantConversationTurn(ConversationTurn):
    """
    Represents a single turn in a conversation by the assistant.
    """

    role: MessageRole = Field(
        description="The role of the message.",
        default=MessageRole.ASSISTANT,
        frozen=True,
    )
    tool_requests: List[ToolRequest] = Field(
        description="The tool requests; only if this is an assistant turn.",
        default=None,
    )

    def run_tools(self):
        for tool_request in self.tool_requests:
            tool_request.run_tool()


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
class PromptBrief(LanguageActionBrief):
    """
    A brief for a completion language action.
    """

    prompt: Union[str, Message] = Field(
        description="The prompt, either plain text or a structured message."
    )


#################################################
class ConversationBrief(LanguageActionBrief):
    """
    A brief for a chat language action.
    """

    conversation: List = Field(
        description="The conversation history."
    )

    def has_pending_tool_calls(self):
        return (
            len(self.conversation) > 0
            and self.conversation[-1].role == MessageRole.ASSISTANT
            and self.conversation[-1].tool_requests
            and len(self.conversation[-1].tool_requests) > 0
        )

    def extend_conversation(self, turn: ConversationTurn, run_tools: bool = True):
        self.conversation.append(turn)

        if self.has_pending_tool_calls() and run_tools:
            self.extend_conversation(
                UserConversationTurn(
                    tool_results=[
                        ToolResponse(
                            request_id=tool_request.request_id,
                            tool_name=tool_request.tool_name,
                            tool_result=tool_request.run_tool().result,
                        )
                        for tool_request in self.conversation[-1].tool_requests
                    ]
                )
            )


#################################################
class CompletionResult(ActionResult):
    """
    Represents the result of a completion action.
    """

    model_name: str = Field(description="The name of the LLM used for completion.")
    input_tokens: Optional[int] = Field(
        description="The number of tokens in the input prompt.", default=None
    )
    output_tokens: Optional[int] = Field(
        description="The number of tokens in the output completion.", default=None
    )

    class Config:
        protected_namespaces = ()

    conversation_turn: object = Field(description="The response turn in the conversation.")


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
            brief=ConversationBrief,
            action=cls,
            result=CompletionResult,
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
            brief=PromptBrief,
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
            ConversationBrief(
                model_name=brief.model_name,
                tools=brief.tools,
                conversation=[
                    UserConversationTurn(
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
                ConversationBrief(
                    model_name=brief.model_name,
                    conversation=[
                        UserConversationTurn(
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
