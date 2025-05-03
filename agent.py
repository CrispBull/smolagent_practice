from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, GradioUI, GoogleSearchTool, VisitWebpageTool, \
    OpenAIServerModel, ToolCallingAgent

from myagents_tools.my_multi_agent import calculate_average_travel_time
from myagents_tools.my_tools import get_current_time_in_timezone, suggest_menu, SuperheroPartyThemeTool, \
    summarize_topic, catering_service_tool, space_image_tool, party_planning_retriever
from myagents_tools.final_answer import final_answer
from myagents_tools.my_tools import check_reasoning_and_plot, images
from dotenv import load_dotenv
import os
from PIL import Image

load_dotenv()
token = os.getenv("HF_TOKEN")

model = HfApiModel(
    model_id="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud/",
    token=token,
    max_tokens=1500,
    temperature=0.5
)

multi_agent_model = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together"
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

multi_agent = CodeAgent(
    model=multi_agent_model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_average_travel_time],
    additional_authorized_imports=["pandas"],
    max_steps=20
)

web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(), VisitWebpageTool(), calculate_average_travel_time
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10
)

web_agent_tool = ToolCallingAgent(
    model=model,
    tools=[GoogleSearchTool(), VisitWebpageTool()],
    name="web_agent",
    max_steps=10,
    description="Browses the web to find information",
)

manager_agent = CodeAgent(
    model=HfApiModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=16192),
    tools = [calculate_average_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15
)

vlm_agent = CodeAgent(
    tools=[],
    model=OpenAIServerModel(model_id="gpt-4o", max_tokens=16192),
    max_steps=20,
    verbosity_level=2
)


if __name__ == "__main__":
    task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
    Also give me some supercar factories with the same cargo plane transfer time."""
    # result = multi_agent.run(task)
    # agent.planning_interval = 4
    GradioUI(agent).launch()
    response = vlm_agent.run(
        """
            Describe the costume and makeup that the comic character in these photos is wearing and return the description.
            Tell me if the guest is The Joker or Wonder Woman.
            """,
        images=images
    )
    # or Just push to HuggingFace Hub
#     managed_agent_prompt = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
# Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
# Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!
#
# Here's an example of how to plot and return a map:
# import plotly.express as px
# df = px.data.carshare()
# fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
#      color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
# fig.show()
# fig.write_image("saved_image.png")
# final_answer(fig)
#
# Never try to process strings using code: when you have a string to read, just print it and you'll see it."""
#     manager_agent.run(managed_agent_prompt)
#     manager_agent.python_executor.state["fig"]

