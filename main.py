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

    def _run_one(self, brief: WeatherBrief):
        return langcraft.ActionResult(result="pretty darn hot at 37 C")


langcraft.Actions.register(WeatherTool.get_descriptor())

if __name__ == "__main__":
    action = langcraft.ChatAction()
    brief = langcraft.ConversationBrief(
        model_name=langcraft.LLMs.get_model_name("gemi"),
        tools=[WeatherTool.NAME],
        conversation=[
            langcraft.UserConversationTurn(
                message=langcraft.Message(text="What is the temperature in New York?"),
            )
        ],
    )

    result = action.run(brief)
    brief.extend_conversation(result.conversation_turn)

    print(brief.model_dump_json(indent=2, exclude_none=True))
