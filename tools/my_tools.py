from smolagents import tool, load_tool, TransformersModel, Tool
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


def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return best_service

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea"""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (eg., 'classic heroes', 'villian masquerade', 'futuristic "
                           "Gotham')."
        }
    }
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(),
                          "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")


image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)



