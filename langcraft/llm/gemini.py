import os
from typing import List
import vertexai
from vertexai.generative_models._generative_models import Part, Content
from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
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

    def _run_one(self, brief: ChatBrief) -> ChatResult:
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
            for image in turn.message.images:
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
            messages, generation_config=generation_config, stream=False
        )

        # get text response
        text_response = response.text.strip(" \n")

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
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )


#################################################
ChatAction.register_implementation(["gemini*"], GeminiChatAction)
