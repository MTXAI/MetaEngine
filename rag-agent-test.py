import os

from smolagents import OpenAIServerModel, ToolCallingAgent

from core.database.chroma import clean_db, create_db, load_db
from engine.agent.tools.rag import RetrieverTool

clean_db()
retriever_tool = RetrieverTool(create_db("data/my_files"))
# retriever_tool = RetrieverTool(load_db())

# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'  # You can change this to your preferred VLM model
model = OpenAIServerModel(model_id="qwen2.5-vl-72b-instruct", api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")

agent = ToolCallingAgent(
    tools=[retriever_tool],
    model=model
)

agent_output = agent.run("MetaEngine是什么？")


print("Final output:")
print(agent_output)