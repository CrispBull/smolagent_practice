from smolagents import CodeAgent, HfApiModel, GradioUI, DuckDuckGoSearchTool
from tools.my_tools import get_current_time_in_timezone, summarize_topic
from tools.final_answer import final_answer
from dotenv import load_dotenv
import os
import yaml

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
    tools=[final_answer, get_current_time_in_timezone, DuckDuckGoSearchTool(), summarize_topic],
    max_steps=6,
    #prompt_templates=prompt_templates # not using a custom template
)

if __name__ == "__main__":
    result = agent.run("What is currently happening in Epigenetics")
    print("Agent execution completed.")