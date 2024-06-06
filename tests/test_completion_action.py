import langcraft

models_to_test = ["claude-3-haiku", "gpt-4o"]


def test_simple_completion():
    for model in models_to_test:
        brief = langcraft.CompletionBrief.from_prompt(
            prompt="Write a haiku about the weather",
            model_name=langcraft.LLMs.resolve_model(model),
        )

        result = langcraft.CompletionAction().run(brief)

        assert result.success
        assert result.model_name.startswith(model)
        assert result.conversation_turn
        assert result.conversation_turn.role == langcraft.MessageRole.ASSISTANT
        assert len(result.conversation_turn.message.text) > 0
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.latency > 0



def test_vision_completion():
    for model in models_to_test:
        brief = langcraft.CompletionBrief.from_prompt(
            prompt=langcraft.Message(
                text="What shape is shown in the image?",
                images=[
                    langcraft.Image.from_file("tests/data/shape.png"),
                ],
            ),
            model_name=langcraft.LLMs.resolve_model(model),
        )

        result = langcraft.CompletionAction().run(brief)

        assert result.success
        assert result.model_name.startswith(model)
        assert result.conversation_turn
        assert result.conversation_turn.role == langcraft.MessageRole.ASSISTANT
        assert len(result.conversation_turn.message.text) > 0
        assert "triangle" in result.conversation_turn.message.text.lower()
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.latency > 0


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Obtain weather information for a location.

    Args:
        location (str): The location for which to get the weather.
        unit (str): The unit of temperature to return, either "celsius" or "fahrenheit".
    """
    return f"The temperature in {location} is 29 degrees {unit.capitalize()}."

langcraft.Actions.generate_action(get_weather)


def test_function_call_completion():
    for model in models_to_test:
        brief = langcraft.CompletionBrief.from_prompt(
            prompt="What is the weather in New York?",
            model_name=langcraft.LLMs.resolve_model(model),
            tools=["get_weather"],
        )

        result = langcraft.CompletionAction().run(brief)

        assert result.success
        assert result.model_name.startswith(model)
        assert result.conversation_turn
        assert result.conversation_turn.role == langcraft.MessageRole.ASSISTANT
        assert result.conversation_turn.tool_call_requests
        assert len(result.conversation_turn.tool_call_requests) == 1
        assert result.conversation_turn.tool_call_requests[0].tool_name == "get_weather"
        assert "new york" in result.conversation_turn.tool_call_requests[0].tool_arguments.location.lower()
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.latency > 0
