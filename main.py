import pydantic
import langcraft


def get_weather(location: str) -> str:
    """
    Obtain weather information for
    a location.

    Args:
        location (str): The location for which to get the weather.
    """
    return "pretty darn hot at 37 C"

d=langcraft.Actions.generate_action(get_weather)
print(d.brief.to_schema())

if __name__ == "__main__":
    brief = langcraft.ConversationBrief(
        model_name=langcraft.LLMs.resolve_model("gemi"),
        tools=["get_weather"],
        conversation=[
            langcraft.UserConversationTurn(
                message=langcraft.Message(text="What is the temperature in New York?"),
            )
        ],
    )

    result = langcraft.ChatAction().run(brief)

    brief.extend_conversation(result.conversation_turn)

    print(brief.model_dump_json(indent=2, exclude_none=True))
