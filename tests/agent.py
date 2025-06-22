import __init__

import argparse
import logging
import os

from smolagents import OpenAIServerModel, ToolCallingAgent

from engine.agent.store.chroma import clean_db, create_db
from engine.agent.tools.rag import RetrieverTool
from engine.config import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--docs", type=str, required=False, default="./test_datas/rag_files")

    args = parser.parse_args()

    clean_db(DATABASE_PATH)
    retriever_tool = RetrieverTool(create_db(DATABASE_PATH, args.docs))
    # retriever_tool = RetrieverTool(load_db())
    print(ONE_API_LLM_MODEL.api_base_url)
    # For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620' and also change 'os.environ.get("ANTHROPIC_API_KEY")'  # You can change this to your preferred VLM model
    model = OpenAIServerModel(
        model_id=ONE_API_LLM_MODEL.model_id,
        api_key=ONE_API_LLM_MODEL.api_key,
        api_base=ONE_API_LLM_MODEL.api_base_url,
    )

    agent = ToolCallingAgent(
        tools=[retriever_tool],
        model=model
    )

    agent_output = agent.run("MetaEngine是什么？")

    logging.info("Final output:")
    logging.info(agent_output)
