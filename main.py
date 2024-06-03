import pydantic
import langcraft


class WeatherBrief(langcraft.ActionBrief):
    location: str = pydantic.Field(
        description="The location for which to get the weather.",
        example="New York",
    )


class WeatherTool(langcraft.Action):

    NAME = "get_weather"

    @classmethod
    def get_descriptor(cls) -> langcraft.ActionDescriptor:
        return langcraft.ActionDescriptor(
            name=WeatherTool.NAME,
            description="Obtain weather information for a location.",
            brief=WeatherBrief,
            action=cls,
            result=langcraft.ActionResult,
        )


langcraft.Actions.register(WeatherTool.get_descriptor())

if __name__ == "__main__":
    action = langcraft.CompletionAction()
    brief = langcraft.CompletionBrief(
        model_name=langcraft.LLMs.get_model_name("claude-3-h"),
        tools=[WeatherTool.NAME],
        prompt=langcraft.Message(
            text="What is the temperature in New York?",
        ),
    )

    result = action.run(brief)

    print(result.model_dump_json(indent=2))
