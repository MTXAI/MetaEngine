import os

from langchain import hub
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.human.agent.agents.qwen_agent import create_structured_qwen_chat_agent
from engine.human.agent.agents.structed_chat_agent import create_structured_chat_agent

if __name__ == '__main__':
    llm = ChatOpenAI(
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 填写DashScope base_url
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2-72b-instruct",
        streaming=True,
    )
    agent_executor = create_structured_qwen_chat_agent(llm, tools=[], callbacks=[], use_custom_prompt=True)
    messages = [
        {
            "role": "user",
            "content": "假设你正在带货一本英语单词书, 请简短准确且友好礼貌地回复弹幕问题: 怎么发货?",
        }
    ]
    for chunk in agent_executor.stream({"input": messages[0]["content"]}):
        print(chunk["output"], end="\n", flush=True)

    prompt = hub.pull("hwchase17/structured-chat-agent")  # default prompt
    agent = create_structured_chat_agent(llm=llm, tools=[], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=[], verbose=True, callbacks=[]
    )
    messages = [
        {
            "role": "user",
            "content": "假设你正在带货一本英语单词书, 请简短准确且友好礼貌地回复弹幕问题: 怎么发货?",
        }
    ]
    for chunk in agent_executor.stream({"input": messages[0]["content"]}):
        print(chunk, end="\n", flush=True)