from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, VisitWebpageTool
from tools.my_tools import get_current_time_in_timezone, summarize_topic, image_generation_tool, suggest_menu, \
    catering_service_tool, SuperheroPartyThemeTool
from tools.final_answer import final_answer
from dotenv import load_dotenv
import os
import datetime


load_dotenv()
token = os.getenv("HF_TOKEN")

# Not using a custom prompt template
# try:
#     with open("prompts.yaml", 'r') as f:
#         prompt_templates = yaml.safe_load(f)
# except Exception:
#     prompt_templates = None

model = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=token,
    max_tokens=1500,
    temperature=0.5
)

agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, DuckDuckGoSearchTool(), summarize_topic, image_generation_tool,
           suggest_menu, VisitWebpageTool, catering_service_tool, SuperheroPartyThemeTool()],
    max_steps=10, additional_authorized_imports=['datetime'], verbosity_level=2
    #prompt_templates=prompt_templates # not using a custom template
)


if __name__ == "__main__":
    # print(np.__version__)
    #Make recommendations for songs to play at party in Lagos, Nigeria
    agent.push_to_hub('Justchidi/smolagent_practice')

    # result = agent.run()
    # print(f'Agent execution completed {result}')
    
    
    
    
    