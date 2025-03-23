## Agent Systems Note

Test building agents with smolagents library

### Agents Intro

AI Agents are programs that makes use of an LLM to reason about problems based on observations and then take actions.

### Re-Act

### Agentic Frameworks

These are frameworks for building ai agents. There are several of these but some common ones are smolagents,
llamaIndex and llamagraph.

While these frameworks are good for building applications around LLMs, they're not always necessary. For simple
cases, like when predefined workflow, a chain of prompt are enough to achieve user requests, then there's no need to
use an agentic framework and you get the advantage of having full control and understanding of the code without the
abstractions that agentic frameworks add. However, when the workflows are complex, like calling functions, using
multiple agents, then we might want to use such frameworks. This would help us with things like;

- providing an LLM to manage the system
- tools the agent can access
- a parser for extracting tool calls from LLM output
- system prompt synced with parser
- memory system
- error logging and retry mechanism

### Smolagents

This is a agent framework that's small with minimal code complexity and abstractions. It works with any LLM via
HuggingFace, external APIs or local models like via ollama. It has first class support for Code Agents where actions
are written in code (python) unlike Json/txt like several other agents. It also has integration with HuggingFace Hub
so that its easy to use Gradio Spaces as tools. Smolagents is useful when you want something that's lightweight,
your application logic isn't complex and you want to quickly experiment without complex configuration.

Agents in smolagents operate as multi-step agents which means that the agents can act in multiple steps where each
step consists of one thought, one tool call for the thought and the execution with the tool. Both CodeAgent
(actions/actions specified in code) and ToolCallingAgent (tool/actions specified in JSON) are subclasses
are of MultiStepAgent. We can define a tool using the `@tool` decorator or the `Tool` class

#### Model integrations

Smolagents can work with many LLM models as long as they
fulfil [certain requirements](https://huggingface.co/docs/smolagents/main/en/reference/models). There are some
predefined classes that helps with model integratons;

- TransformersModel - for a local transformer pipeline
- HfApiModel - for serverless inference calls using HuggingFace's infra or third party inference providers
- LiteLLMModel - for lightweight model interactions
- OpenAIServerModel - for connecting to any service that offers an OpenAI API interface
- AzureOpenAIServerModel - for integration to Azure OpenAI deployments

#### Code Agents

Code agents in smolagents (CodeAgent) generate python code for the action they have to perform. This approach helps
reduce the amount of parsing required, reduces the amount of required actions and enables the reuse of code
functions. Writing actions in code offers advantages such as being able to do anything that is computationally
possible, working directly with complex objects such as images, combine and reuse actions, and code is already
natural with LLMs.
`CodeAgent` performs actions via a cycle of steps. First the system prompt is stored in the `SystemPromptStep` and
the user query is logged in a `TaskStep`. Then we enter a loop where we first write the agent's log into a list of
LLM-readable chat messages using the `agent.write_memory_to_messages()` method, these messages are then sent to a
model which generates a completion, the completion is parsed to extract the action, which given its a code agent
would be a code snippet. The extracted action executed and the result logged into memory in an`ActionStep`. At the
end of each step, if the agent includes any function calls in the `agent.step_callback` callback that function is
executed.

Due to security reasons, smolagents python code snippets are executed in
a [secure sandboxed environment](https://huggingface.co/docs/smolagents/tutorials/secure_code_execution) within the
framework. Imports outside a predefined safe list are blocked by default, however we can authorize it to use
additional imports by passing those imports as strings in the `additional_authorized_imports`. For example

```python
agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes
    
    If we start right now, at what time will the party be ready?
    """
)
```

#### Tool Calling Agent
`ToolCallingAgent` is the other alternative to `CodeAgent` available in smolagents framework. This agent uses built 
in tool calling capabilities of LLM providers to generate tool calls as JSON. This is the standard approach used by 
OpenAI, Anthropic, etc. For example, for an action to search for catering services and party ideas, a `CodeAgent` 
would create an action like;
```python
for query in [
    "Best catering services in Lagos",
    "Party theme ideas for superheroes"
]:
    print(web_search(f"Search for {query}"))
```
On the other hand, a `ToolCallingAgent` would instead create something like;
```json
[
  {"name": "web_search", "arguments": "Best catering services in Gotham City" },
  {"name": "web_search", "arguments": "Party theme ideas for superheroes" }
]
```
This json is then used to execute the tool calls. ToolCallingAgent internally work similar to CodeAgent with the 
multi-step workflow, however they differ in how they structure their action as shown above which the system then 
parses to execute the right tools.

### Tools
Tools are functions that the LLM can call within an agent system. The interface for a tool has the following components;
- Name: The name of the tool
- Description: What the tool does
- Input types and description: The arguments which the tool function accepts and their description
- Output type: what the tool returns
For example, check out the following summary tool;
```python
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
```
In smolagents, we can create a tool by extending the `Tool` class or using the `@tool` decorator. In the above the 
tool name is `summarize_topic`, the tool description is in the function docstring, the input type is in the function 
argument and description is in the docstring, the output type is the function return type. It's recommended to name 
your function properly and write good descriptions for the function as well as for both input ad output. 

We can also define a tool with a class;
```python
from smolagents import Tool

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
```
In the above, we can see it extends the `Tool` class. This approach is usually recommended for more complex tools. 
In this class we can see we have defined the `name`, `output_type`, `description`, `inputs` and then a `forward` 
function which is the method containing the inference logic to execute. We can look at the `Tool` class definition 
to learn more about these overridden properties. 

Smolagent comes with a set of default tools, you can find these in the `default_tools.py` file.

### Agentic RAG Systems
Traditional RAG (Retrieval Augumented Generation) systems basically combine data retrieval and gen ai to provide 
context aware responses. For example, given a user query, we pass the query to a search engine or db for example and 
retrieve results, this retrieved result is then given to a model alongside the query and then the model generates a 
response based on the query and the earlier retrieved information. Agentic RAG extends this by combining autonomous 
agents with dynamic knowledge retrieval via an intelligent control of both the retrieval and generation process. 
Traditional RAG systems has the limitation of relying on a single retrieval step and focusing on direct semantic 
similarity which may overlook certain relevant information, Agentic RAG addresses this by allowing the agent to 
formulate the search queries, critique the retrieved results and conduct multiple retrieval steps if necessary.

For example, say we enter the following query in an agentic rag system; *"Search for luxury superhero-themed party 
ideas, including decorations, entertainment, and catering."*, assuming we're using the `DuckDuckSearchTool` in 
smolagents, our agent would first analyze the request to identify key elements of the query, then it performs 
retrieval, in this case using our search tool, then it synthesizes the information after gathering the search 
results, then it stores the result for future references in case it needs it later for subsequent tasks. 

Sometimes we might need a custom knowledge base for certain tasks. Such data are usually stored in a vector database.
A vector database is a database for storing, managing and searching numerical representations (embeddings) of text 
or other kinds of data. This approach of saving data enables semantic search by identifying similar data points 
based on their numerical representation in this high dimensional space. An simple example of this is the 
`PartyPlanningRetrievalTool` defined in the `my_tools.py` file.

When building agentic RAG systems, the agent can use strategies like;
- Query reformulation: Instead of using the same raw user query, the agent can craft optimized search terms to 
  better match the target documents
- Use multi-step retrieval so that initial results are used to inform subsequent queries
- combine information from multiple sources like web and local documentation
- explore ways to validate results for relevance and accuracy before being included in responses. 

Building effective agentic RAG systems requires carefully considering all the key aspects as well as the tools made 
available to it based on type of query and context. Memory systems in these programs helps maintain conversation 
history to avoid repetitive retrievals. Its important to also have fallback strategies to ensure that the systems 
can still provide some value when the primary retrieval methods fail. 


### Sharing and using tools from community
We can share our agent with the community via Huggingface Hub too and anyone can easily download and use the agent
directly from the hub. To do this;

```python
agent.push_to_hub("yourHFUserName/RepoName")
```

To download the agent again;
```python
my_agent = agent.from_hub("yourHFUserName/RepoName", trust_remote_code=True)
my_agent.run("Give me a playlist for a birthday party")
```
Shared agents are also available as Hugging Face Spaces, so we can interact with them in real time. We can also 
import a tool from Huggingface Hub instead of building from scratch. For example, we can import an image generation 
tool from huggingface hub and make use of it in our agent;
```python
from smolagents import load_tool

image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
```
We can also import a huggingface space as a tool using the `Tool.from_space()` function, making it possible to 
integrate with thousands of spaces from the community for various tasks. 

We can also import tools from Langchain in our smolagents workflow. To do this, we'll make use of the `Tool.
from_langchain()`

#### Logging and Monitoring
Smolagents uses OpenTelemetry standard for instrumenting agent runs, making it possible to inspection of agent
activities and logging. Using [Langfuse](https://langfuse.com/)
or [alternatives](https://huggingface.co/docs/smolagents/tutorials/inspect_runs) and `SmolagentsInstrumentator` we
can add the ability to track and analyze our agents behavior. To do this, we first install the necessary dependencies;

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

Then we go to Langfuse to create an account and get our API keys, after which we can now do the following;
```python
import os
import base64
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

LANGFUSE_PUBLIC_KEY="pk-1f-...."
LANGFUSE_SECRET_KEY="pk-1f-...."
LANGFUSE_AUTH=base64.b16encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel" # For EU data region
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://us.cloud.langfuse.com/api/public/otel" # US data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
SmolagentsInstrumentor().instrument(trace_provider=trace_provider)
```

With the above, runs from our agent are now being logged to Langfuse, giving us full visibility into the agents 
behavior.

#### Classwork
- Play with other peoples agents, even import 
- Connect to OpenTelemetry and see own agent in action
- Setup Gradio UI 
- Answer questions in course forum
- 