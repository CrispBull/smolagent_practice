from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, GradioUI
from myagents_tools.my_tools import get_current_time_in_timezone, suggest_menu, SuperheroPartyThemeTool, \
    summarize_topic, catering_service_tool, space_image_tool, party_planning_retriever
from myagents_tools.final_answer import final_answer
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv("HF_TOKEN")

model = HfApiModel(
    model_id="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud/",
    token=token,
    max_tokens=1500,
    temperature=0.5
)

# Create a simplified agent for pushing to HuggingFace with fewer tools
# to avoid errors with external dependencies
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, DuckDuckGoSearchTool(),
           suggest_menu, SuperheroPartyThemeTool(), catering_service_tool,
           summarize_topic, space_image_tool, party_planning_retriever],
    max_steps=10,
    additional_authorized_imports=['datetime'],
    verbosity_level=2
)

if __name__ == "__main__":
    GradioUI(agent).launch()
    # Just push to HuggingFace Hub
