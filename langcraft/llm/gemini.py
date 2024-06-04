import os
from typing import List, Dict
import vertexai
from vertexai.generative_models._generative_models import Part, Content
from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
    FunctionDeclaration,
    Tool,
)
from langcraft.llm.llm_action import (
    LanguageAction,
    ConversationBrief,
    CompletionResult,
    ChatAction,
    Message,
    MessageRole,
    AssistantConversationTurn,
    ToolRequest,
    Actions,
)


#################################################
class VertexClient:
    """A class that represents the Vertex client."""

    _initialized = False

    @classmethod
    def init(cls):
        """
        Initializes the Vertex client.
        """
        if cls._initialized is False:
            VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT")
            VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION")

            vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)

            cls._initialized = True


#################################################
class GeminiChatAction(LanguageAction):
    """
    A chat action that uses Gemini to generate chats.
    """

    def _compile_tools(self, tool_names: List[str]) -> Tool:
        """
        Compile a list of tools into a list of dictionaries.

        Args:
            tools (List[str]): A list of names of tools.

        Returns:
            Tool: A tool object.

        """
        return [
            Tool(
                [
                    FunctionDeclaration(
                        name=action_descriptor.name,
                        description=action_descriptor.description,
                        parameters=action_descriptor.brief.to_schema(),
                    )
                    for action_descriptor in list(
                        map(lambda tool_name: Actions.get(tool_name), tool_names)
                    )
                ]
            )
        ]

    def _run_one(self, brief: ConversationBrief) -> CompletionResult:
        """
        Executes the Chat action.

        Args:
            brief (ChatBrief): The brief information about the Chat action.

        Returns:
            ChatResult: The result of the Chat action execution.
        """

        # set up the model
        VertexClient.init()
        model = GenerativeModel(
            model_name=brief.model_name,
            system_instruction=[brief.system] if brief.system else None,
        )

        # compile messages
        messages = []

        for turn in brief.conversation:
            parts = []
            for image in turn.message.images or []:
                parts.append(
                    Part.from_data(data=image.image_data, mime_type=image.mime_type)
                )

            if turn.message.text:
                parts.append(Part.from_text(turn.message.text))

            content = Content(
                parts=parts, role="user" if turn.role == MessageRole.USER else "model"
            )

            messages.append(content)

        # set up params
        generation_config = GenerationConfig(
            temperature=brief.temperature,
            max_output_tokens=brief.max_tokens,
            stop_sequences=[brief.stop] if brief.stop else None,
        )

        # get completion
        response = model.generate_content(
            messages,
            generation_config=generation_config,
            tools=self._compile_tools(brief.tools),
            stream=False,
        )

        # parse response
        text_response = None
        tool_requests = []

        for response_element in response.candidates:
            if response_element.function_calls:
                for tool_call in response_element.function_calls:
                    tool_requests.append(
                        ToolRequest(
                            request_id=tool_call.name,
                            tool_name=tool_call.name,
                            tool_arguments=Actions.create_brief(
                                tool_call.name, tool_call.args
                            ),
                        )
                    )
            else:
                text_response = response_element.text.strip(" \n")

        turn = AssistantConversationTurn(
            message=Message(text=text_response) if text_response else None,
            tool_requests=tool_requests,
        )

        # return result
        return CompletionResult(
            model_name=brief.model_name,
            result=text_response or "",
            conversation_turn=turn,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )


#################################################
ChatAction.register_implementation(["gemini*"], GeminiChatAction)
