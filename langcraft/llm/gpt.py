import os
from typing import List, Dict
import json
import httpx
import openai
import logging
from langcraft.action import ActionBrief, ActionResult
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
from langcraft.llm.llm_embedding import (
    EmbeddingDelegateAction,
    EmbeddingResult,
    EmbeddingAction,
)
import tiktoken


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
            if os.environ.get("AZURE_OPENAI_API_KEY", "").strip() != "":
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
class GPTCompletionAction(CompletionDelegateAction):
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
        tools = [
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

        return tools if len(tools) > 0 else None

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

        if brief.system:
            messages.append(
                {
                    "role": "system",
                    "content": brief.system,
                }
            )

        for turn in brief.conversation:
            content = []

            if turn.message:
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

            if turn.role == MessageRole.ASSISTANT:
                tool_call_requests = []
                for tool_call_request in turn.tool_call_requests or []:
                    tool_call_requests.append(
                        {
                            "id": tool_call_request.request_id,
                            "type": "function",
                            "function": {
                                "name": tool_call_request.tool_name,
                                "arguments": json.dumps(
                                    tool_call_request.tool_arguments.dict()
                                ),
                            },
                        }
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_call_requests,
                    }
                )
            elif turn.role == MessageRole.USER:
                if len(content) > 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": content,
                        }
                    )

                for tool_call_result in turn.tool_call_results or []:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_result.request_id,
                            "content": tool_call_result.tool_result,
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
        tool_call_requests = []

        for response_element in response.choices:
            if response_element.message.content:
                text_response = response_element.message.content.strip(" \n")
            for tool_call_request in response_element.message.tool_calls or []:
                tool_call_requests.append(
                    ToolCallRequest(
                        request_id=tool_call_request.id,
                        tool_name=tool_call_request.function.name,
                        tool_arguments=Actions.create_brief(
                            tool_call_request.function.name,
                            json.loads(tool_call_request.function.arguments),
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
            result=text_response or "",
            conversation_turn=turn,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )


#################################################
CompletionAction.register_implementation(["gpt*"], GPTCompletionAction)


#################################################
class GPTEmbeddingAction(EmbeddingDelegateAction):

    def __init__(self):
        super().__init__(max_batch_size=20)

    def _run_one_batch(self, briefs: List[ActionBrief]) -> List[ActionResult]:
        """
        Executes the action on a batch of briefs.

        Args:
            briefs (List[ActionBrief]): The list of action briefs.

        Returns:
            List[ActionResult]: The list of action results.
        """
        # compile the texts
        texts = [brief.text.replace("\n", " ").strip() for brief in briefs]

        # obtain embeddings
        client = OpenAIClient.get()

        model_name = briefs[0].model_name
        dimensions = briefs[0].dimensions or None

        response = client.embeddings.create(
            input=texts,
            model=model_name,
            dimensions=dimensions or openai.NotGiven(),
        )

        # parse response
        results = []
        encoding = tiktoken.get_encoding("cl100k_base")
        for response_element, text in zip(response.data, texts):
            results.append(
                EmbeddingResult(
                    model_name=model_name,
                    embedding=response_element.embedding,
                    tokens=len(encoding.encode(text)),
                )
            )

        return results


#################################################
EmbeddingAction.register_implementation(["text-*"], GPTEmbeddingAction)
