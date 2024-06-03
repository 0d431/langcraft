import os
from typing import List
import httpx
import openai
import logging
from langcraft.action import ActionDescriptor
from langcraft.llm.llm_action import (
    LanguageAction,
    ChatBrief,
    ChatResult,
    ChatAction,
    Message,
    MessageRole,
    ConversationTurn,
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

        if brief.system:
            messages.append(
                {
                    "role": "system",
                    "content": brief.system,
                }
            )

        for turn in brief.conversation:
            content = []
            for image in turn.message.images:
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
            temperature=brief.temperature,
            max_tokens=brief.max_tokens,
            stop=brief.stop or openai.NotGiven(),
            seed=42,
        )

        # get text response
        text_response = response.choices[0].message.content.strip(" \n")

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
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


#################################################
ChatAction.register_implementation(["gpt*"], GPTChatAction)
