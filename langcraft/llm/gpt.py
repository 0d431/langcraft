import os
from typing import List, Dict
import json
import httpx
import openai
import logging
from langcraft.action import ActionDescriptor
from langcraft.llm.llm_action import (
    LanguageAction,
    ConversationBrief,
    CompletionResult,
    ChatAction,
    Message,
    MessageRole,
    ConversationTurn,
    AssistantConversationTurn,
    ToolRequest,
    Actions,
)


#################################################
class OpenAIClient:
    """A class that represents the OpenAI client."""

    _openai_client = None

    @classmethod
    def get(cls):
        """
        Returns the OpenAI client instance.

        If the client instance is not already created, it will be created based on the environment variables.
        If the environment variable 'AZURE_OPENAI_API_KEY' is set, an instance of 'AzureOpenAI' will be created.
        Otherwise, an instance of 'OpenAI' will be created.

        Returns:
            The OpenAI client instance.

        Raises:
            KeyError: If the required environment variables are not set.
        """
        if cls._openai_client is None:
            # set up logging of httpx
            httpx_logger = logging.getLogger("httpx")
            httpx_logger.setLevel(logging.ERROR)

            # OAI or Azure?
            if os.environ["AZURE_OPENAI_API_KEY"].strip() != "":
                cls._openai_client = openai.AzureOpenAI(
                    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                    max_retries=5,
                    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    timeout=httpx.Timeout(60.0, connect=3.0),
                )
            else:
                cls._openai_client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    max_retries=5,
                    timeout=httpx.Timeout(60.0, connect=3.0),
                )

        return cls._openai_client


#################################################
class GPTChatAction(LanguageAction):
    """
    A chat action that uses GPT to generate chats.
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
                "type": "function",
                "function": {
                    "name": action_descriptor.name,
                    "description": action_descriptor.description,
                    "parameters": action_descriptor.brief.to_schema(),
                },
            }
            for action_descriptor in list(
                map(lambda tool_name: Actions.get(tool_name), tool_names)
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
        # compile messages
        messages = []

        if brief.system:
            messages.append(
                {
                    "role": "system",
                    "content": brief.system,
                }
            )

        for turn in brief.conversation:
            content = []
            for image in turn.message.images or []:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.mime_type};base64,{image.image_data}"
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

        # obtain completion
        client = OpenAIClient.get()
        response = client.chat.completions.create(
            model=brief.model_name,
            messages=messages,
            tools=self._compile_tools(brief.tools),
            temperature=brief.temperature,
            max_tokens=brief.max_tokens,
            stop=brief.stop or openai.NotGiven(),
            seed=42,
        )

        # parse response
        text_response = None
        tool_requests = []

        for response_element in response.choices:
            if response_element.message.content:
                text_response = response_element.message.content.strip(" \n")
            if response_element.message.tool_calls is not None:
                for tool_call in response_element.message.tool_calls:
                    tool_requests.append(
                        ToolRequest(
                            request_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            tool_input=Actions.create_brief(
                                tool_call.function.name,
                                json.loads(tool_call.function.arguments),
                            ),
                        )
                    )

        turn = AssistantConversationTurn(
            message=Message(text=text_response) if text_response else None,
            tool_requests=tool_requests,
        )

        # return result
        return CompletionResult(
            model_name=brief.model_name,
            result=text_response or "<tool call>",
            conversation_turn=turn,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


#################################################
ChatAction.register_implementation(["gpt*"], GPTChatAction)
