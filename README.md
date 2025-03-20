from smolagents import HfApiModelfrom smolagents import CodeAgent

# Agent Systems Note

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

We can share our agent with the community via Huggingface Hub too and anyone can easily download and use the agent 
directly from the hub. To do this;
```python
agent.push_to_hub("yourHFUserName/RepoName")
```
