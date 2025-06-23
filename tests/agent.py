import __init__

import argparse
import logging
import os

from smolagents import OpenAIServerModel

from engine.agent.agents.smol.agents import QaAgent
from engine.agent.model.history import History
from engine.agent.vecdb.chroma import clean_db, create_db
from engine.agent.tools.rag import RetrieverTool
from engine.config import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--docs", type=str, required=False, default="./test_datas/rag_files")

    args = parser.parse_args()

    vecdb_path = DEFAULT_PROJECT_CONFIG.vecdb_path
    clean_db(vecdb_path)
    retriever_tool = RetrieverTool(create_db(vecdb_path, args.docs))
    # retriever_tool = RetrieverTool(load_db())
    print(ONE_API_LLM_MODEL.api_base_url)
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

    h = []
    user_history = History("user", "MetaEngine能用来做饭吗")
    ai_history = History("ai", "MetaEngine不能用来做饭吃")
    h.append(user_history)
    h.append(ai_history)

    agent = QaAgent(retriever_tool, model, history=History.convert_histories_to_msg_str(h))
    agent_output = agent.run("MetaEngine是什么？能用来做饭吃吗")

    logging.info("Final output:")
    logging.info(agent_output)
