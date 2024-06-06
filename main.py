import random
import langcraft


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Obtain weather information for
    a location.

    Args:
        location (str): The location for which to get the weather.
        unit (str): The unit of temperature to return, either "celsius" or "fahrenheit".
    """
    return f"The temperature in {location} is {random.randrange(-10,30)} degrees {unit.capitalize()}."


langcraft.Actions.generate_action(get_weather)

brief = langcraft.CompletionBrief.from_prompt(
    prompt="Write a peom",
    model_name=langcraft.LLMs.resolve_model("gpt-4o"),
    tools=["get_weather","search_web_jina"],
)

result = langcraft.CompletionAction().run_with_tools(brief)
print(result.model_dump_json(indent=2, exclude_none=True))
