import os
from typing import List, Dict
import httpx
import anthropic
from langcraft.llm.llm_completion import (
    CompletionBrief,
    CompletionAction,
    CompletionResult,
    Message,
    MessageRole,
    AssistantConversationTurn,
    ToolCallRequest,
    Actions,
    CompletionDelegateAction,
)


#################################################
class AnthropicClient:
    """A class that represents the Anthropic client."""

    _anthropic_client = None

    @classmethod
    def get(cls):
        """
        Returns the Anthropic client instance.

        If the client instance is not already created, it will be created based on the environment variables.

        Returns:
            The client instance.

        Raises:
            KeyError: If the required environment variables are not set.
        """
        if cls._anthropic_client is None:
            cls._anthropic_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                timeout=httpx.Timeout(60.0, connect=3.0),
            )

        return cls._anthropic_client


#################################################
class ClaudeCompletionAction(CompletionDelegateAction):
    """
    A chat action that uses Claude to generate chats.
    """

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

    def _run_one(self, brief: CompletionBrief) -> CompletionResult:
        """
        Executes the Chat action.

        Args:
            brief (ChatBrief): The brief information about the Chat action.

        Returns:
            ChatResult: The result of the Chat action execution.
        """
        # compile messages
        messages = []
        for turn in brief.conversation:
            content = []
            if turn.message:
                for image in turn.message.images or []:
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image.mime_type,
                                "data": image.image_data,
                            },
                        }
                    )
                content.append({"type": "text", "text": turn.message.text})

            if turn.role == MessageRole.ASSISTANT:
                for tool_call_request in turn.tool_call_requests or []:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call_request.request_id,
                            "name": tool_call_request.tool_name,
                            "input": tool_call_request.tool_arguments.dict(),
                        }
                    )
            elif turn.role == MessageRole.USER:
                for tool_call_result in turn.tool_call_results or []:
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_result.request_id,
                            "content": tool_call_result.tool_result,
                        }
                    )

            messages.append(
                {
                    "role": "user" if turn.role == MessageRole.USER else "assistant",
                    "content": content,
                }
            )

        # obtain completion
        client = AnthropicClient.get()
        response = client.messages.create(
            model=brief.model_name,
            system=brief.system if brief.system else "",
            messages=messages,
            tools=self._compile_tools(brief.tools),
            temperature=brief.temperature,
            max_tokens=brief.max_tokens,
            stop_sequences=[brief.stop] if brief.stop else None,
        )

        # parse response
        text_response = None
        tool_call_requests = []

        for response_element in response.content:
            if response_element.type == "text":
                text_response = response_element.text.strip(" \n")
            elif response_element.type == "tool_use":
                tool_call_requests.append(
                    ToolCallRequest(
                        request_id=response_element.id,
                        tool_name=response_element.name,
                        tool_arguments=Actions.create_brief(
                            response_element.name, response_element.input
                        ),
                    )
                )

        turn = AssistantConversationTurn(
            message=Message(text=text_response) if text_response else None,
            tool_call_requests=(
                tool_call_requests if len(tool_call_requests) > 0 else None
            ),
        )

        # return result
        return CompletionResult(
            model_name=brief.model_name,
            conversation_turn=turn,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


#################################################
CompletionAction.register_implementation(["claude*"], ClaudeCompletionAction)
