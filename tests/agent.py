import __init__

import argparse
import logging
import os
from langchain_openai import ChatOpenAI
from smolagents import OpenAIServerModel, FinalAnswerStep

from engine.agent.agents.custom.agents import KnowledgeAgent
from engine.agent.agents.smol.agents import QaAgent
from engine.agent.model.history import History
from engine.agent.tools.rag import RetrieverTool
from engine.agent.vecdb.chroma import clean_db, create_db
from engine.config import *

def test_qa_agent():
    global vecdb_path
    vecdb_path = DEFAULT_PROJECT_CONFIG.vecdb_path
    clean_db(vecdb_path)
    retriever_tool = RetrieverTool(create_db(vecdb_path, args.docs))
    # retriever_tool = RetrieverTool(load_db())
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'  # You can change this to your preferred VLM model
    # model = OpenAIServerModel(
    #     model_id=ONE_API_LLM_MODEL.model_id,
    #     api_key=ONE_API_LLM_MODEL.api_key,
    #     api_base=ONE_API_LLM_MODEL.api_base_url,
    # )
    model = OpenAIServerModel(
        model_id=QWEN_LLM_MODEL.model_id,
        api_key=QWEN_LLM_MODEL.api_key,
        api_base=QWEN_LLM_MODEL.api_base_url,
    )
    # model = OpenAIServerModel(
    #     model_id=ONE_API_LLM_MODEL.model_id,
    #     api_key=ONE_API_LLM_MODEL.api_key,
    #     api_base=ONE_API_LLM_MODEL.api_base_url,
    # )
    h = []
    user_history = History("user", "MetaEngine的作者是谁?")
    ai_history = History("ai", "MetaEngine作者是爱因斯坦")
    h.append(user_history)
    h.append(ai_history)
    agent = QaAgent(retriever_tool, model, history=History.convert_histories_to_msg_str(h))
    logging.info("Final output:")
    for agent_output in agent.run("MetaEngine是什么? 作者是谁?", stream=True):
        if isinstance(agent_output, FinalAnswerStep):
            # logging.info(agent_output)
            res = str(agent_output.output)
            print(res)

def test_knowledge_agent():
    global vecdb_path
    vecdb_path = DEFAULT_PROJECT_CONFIG.vecdb_path
    clean_db(vecdb_path)
    vector_store = create_db(vecdb_path, args.docs)
    # retriever_tool = RetrieverTool(load_db())
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'  # You can change this to your preferred VLM model
    # model = OpenAIServerModel(
    #     model_id=ONE_API_LLM_MODEL.model_id,
    #     api_key=ONE_API_LLM_MODEL.api_key,
    #     api_base=ONE_API_LLM_MODEL.api_base_url,
    # )
    model = ChatOpenAI(
        model=QWEN_LLM_MODEL.model_id,
        api_key=QWEN_LLM_MODEL.api_key,
        base_url=QWEN_LLM_MODEL.api_base_url,
    )
    # model = OpenAIServerModel(
    #     model_id=ONE_API_LLM_MODEL.model_id,
    #     api_key=ONE_API_LLM_MODEL.api_key,
    #     api_base=ONE_API_LLM_MODEL.api_base_url,
    # )
    h = []
    user_history = History("user", "MetaEngine的作者是谁?")
    ai_history = History("ai", "MetaEngine作者是爱因斯坦")
    h.append(user_history)
    h.append(ai_history)
    agent = KnowledgeAgent(model, vector_store, History.convert_histories_to_msg_str(h))
    print("Final output:")
    for output in agent.query("MetaEngine是什么? 作者是谁?"):
        print(output.content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--docs", type=str, required=False, default="./test_datas/rag_files")

    args = parser.parse_args()

    # test_qa_agent()
    test_knowledge_agent()
