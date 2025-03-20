from smolagents import tool, load_tool, TransformersModel
from huggingface_hub import login


@tool
def get_current_time_in_timezone(timeZone: str) -> str:
    """
    Returns the current local time for a given timezone.

    Args:
        timeZone: A valid timezone string (e.g., 'America/New_York')
    """
    import pytz, datetime
    try:
        tz = pytz.timezone(timeZone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current time in {timeZone} is {local_time}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def summarize_topic(topic: str) -> str:
    """
    This agent uses a web search tool to getch search results for the provided topic and the summarizes them and
    returns the summary

    Args:
        topic: The topic to provide summary for
    """
    from transformers import pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(topic, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occassion

    Args:
        occasion: The type of occassion for the party. Allowed values are:
            - "Casual": Menu for casual party.
            - "Formal": Menu for formal party.
            - "Superhero": Menu for superhero party.
            - "Custom": Custom menu.
    """
    if occasion.lower() == "casual":
        return "Pizza, snacks, and drinks"
    elif occasion.lower() == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion.lower() == "superhero":
        return "Buffet with high energy and healthy food"
    else:
        return "Custom menu for the butler."


image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)



