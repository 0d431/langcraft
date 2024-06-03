import os
from typing import List
import httpx
import anthropic
from langcraft.action import ActionDescriptor
from langcraft.llm.llm_action import (
    LanguageAction,
    ChatBrief,
    ChatResult,
    ChatAction,
    Message,
    MessageRole,
    ConversationTurn,
    ToolRequest,
    ToolResponse,
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
class ClaudeChatAction(LanguageAction):
    """
    A chat action that uses Claude to generate chats.
    """

    def _run_one(self, brief: ChatBrief) -> ChatResult:
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
            for image in turn.message.images:
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

            messages.append(
                {
                    "role": "user" if turn.role == MessageRole.USER else "assistant",
                    "content": content,
                }
            )

        # compile tools
        tool_descriptions = self._compile_tools(brief.tools)

        # obtain completion
        client = AnthropicClient.get()
        response = client.messages.create(
            model=brief.model_name,
            system=brief.system if brief.system else "",
            messages=messages,
            tools=tool_descriptions,
            temperature=brief.temperature,
            max_tokens=brief.max_tokens,
            stop_sequences=[brief.stop] if brief.stop else None,
        )

        # tool use?
        if response.stop_reason == "tool_use":
            # no text response
            text_response = response.content[0].text.strip(" \n")

            # construct updated message history
            history = brief.conversation.copy()
            history.append(
                ConversationTurn(
                    role=MessageRole.TOOL_REQUEST,
                    message=ToolRequest(
                        request_id=response.content[-1].id,
                        tool_name=response.content[-1].name,
                        tool_input=response.content[-1].input,
                    ),
                )
            )
        else:
            # plain response
            text_response = response.content[0].text.strip(" \n")

            # construct updated message history
            history = brief.conversation.copy()
            history.append(
                ConversationTurn(
                    role=MessageRole.ASSISTANT,
                    message=Message(text=text_response),
                )
            )

        # return result
        return ChatResult(
            model_name=brief.model_name,
            response=text_response,
            conversation=history,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


#################################################
ChatAction.register_implementation(["claude*"], ClaudeChatAction)
