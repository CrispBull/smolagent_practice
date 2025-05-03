import os.path

import helium
from PIL import Image
import requests
from io import BytesIO
from litellm import max_tokens
from smolagents import tool, load_tool, TransformersModel, Tool, OpenAIServerModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents.utils import encode_image_base64, make_image_url


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


@tool
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


class PartyPlanningRetrievalTool(Tool):
    name = "party_planning_retriever"
    description = "uses semantic search to retrieve relevant party planning ideas for Alfred's superhero-themed party at Wayne Manor"
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning or superhero themes."
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.",
     "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.",
     "source": "Entertainment Ideas"},
    {
        "text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'",
        "source": "Catering Ideas"},
    {
        "text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.",
        "source": "Decoration Ideas"},
    {
        "text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.",
        "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

party_planning_retriever = PartyPlanningRetrievalTool(docs_processed)

# image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)


space_image_tool = Tool.from_space("black-forest-labs/FLUX.1-schnell",
                                   name="image_generator",
                                   description="Generate an image from a prompt"
                                   )

# langchain_search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=16192)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under map_saved.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # Joker image
]

images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)







