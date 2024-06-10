import os
from typing import List, Dict, Any
import vertexai
from vertexai.generative_models._generative_models import (
    Part,
    Content,
    gapic_content_types,
    gapic_tool_types,
)
from vertexai.preview.generative_models import (
    GenerativeModel,
    GenerationConfig,
    FunctionDeclaration,
    Tool,
)
from langcraft.llm.llm_completion import (
    CompletionDelegateAction,
    CompletionBrief,
    CompletionResult,
    CompletionAction,
    Message,
    MessageRole,
    AssistantConversationTurn,
    ToolCallRequest,
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
class GeminiCompletionAction(CompletionDelegateAction):
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
        tools = [
            Tool(
                [
                    FunctionDeclaration(
                        name=action_descriptor.name,
                        description=action_descriptor.description,
                        parameters=action_descriptor.brief.to_schema(),
                    )
                ]
            )
            for action_descriptor in list(
                map(lambda tool_name: Actions.get(tool_name), tool_names)
            )
        ]

        return tools if len(tools) > 0 else None

    def _run_one(self, brief: CompletionBrief) -> CompletionResult:
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
            if turn.message:
                parts = []
                for image in turn.message.images or []:
                    parts.append(
                        Part.from_data(data=image.image_data, mime_type=image.mime_type)
                    )

                if turn.message.text:
                    parts.append(Part.from_text(turn.message.text.strip(" \n")))

                if len(parts) > 0:
                    messages.append(
                        Content(
                            parts=parts,
                            role="user" if turn.role == MessageRole.USER else "model",
                        )
                    )

            if turn.role == MessageRole.ASSISTANT:
                for tool_call_request in turn.tool_call_requests or []:
                    messages.append(
                        Content(
                            parts=[
                                Part._from_gapic(
                                    raw_part=gapic_content_types.Part(
                                        function_call=gapic_tool_types.FunctionCall(
                                            name=tool_call_request.tool_name,
                                            args=tool_call_request.tool_arguments.dict(),
                                        )
                                    )
                                )
                            ],
                            role="model",
                        )
                    )
            elif turn.role == MessageRole.USER:
                for tool_call_result in turn.tool_call_results or []:
                    messages.append(
                        Content(
                            parts=[
                                Part.from_function_response(
                                    name=tool_call_result.tool_name,
                                    response={"content": tool_call_result.tool_result},
                                )
                            ]
                        )
                    )

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
        tool_call_requests = []

        for response_element in response.candidates:
            if response_element.function_calls:
                for tool_call_request in response_element.function_calls:
                    tool_call_requests.append(
                        ToolCallRequest(
                            request_id=tool_call_request.name,
                            tool_name=tool_call_request.name,
                            tool_arguments=Actions.create_brief(
                                tool_call_request.name, tool_call_request.args
                            ),
                        )
                    )
            else:
                text_response = response_element.text.strip(" \n")

        turn = AssistantConversationTurn(
            message=Message(text=text_response) if text_response else None,
            tool_call_requests=(
                tool_call_requests if len(tool_call_requests) > 0 else None
            ),
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
CompletionAction.register_implementation(["gemini*"], GeminiCompletionAction)
