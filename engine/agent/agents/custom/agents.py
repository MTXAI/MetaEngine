from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing import Iterator, AsyncGenerator

from engine.agent.agents.base_agent import BaseAgent


class KnowledgeAgent(BaseAgent):
    """
    只与大模型交互一次，先联网搜索和查找rag，将结果放入prompt，最后流式返回query结果
    """

    def __init__(self, model, vector_db):
        self.llm = model
        self.vector_db = vector_db
        self.search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)

    def _retrieve_knowledge(self, query: str) -> str:
        docs = self.vector_db.similarity_search(query, k=3)
        return "\n".join([d.page_content for d in docs])

    def _web_search(self, query: str) -> str:
        try:
            return self.search_wrapper.run(query)
        except Exception:
            return ""

    def _build_prompt(self, question: str, rag: str, web: str, chat_history: str) -> str:
        prompt = f"""
            你是一个专业助手，请根据以下信息回答用户问题。
            
            【内部知识库检索结果】
            {rag}
            
            【网络搜索结果】
            {web}
            
            【历史聊天记录】
            {chat_history}
            
            【用户问题】
            {question}
            
            请结合上述所有信息，给出准确、简明的中文回答。注意知识库检索结果优先级高于网络搜索结果，网络搜索结果优先级高于历史聊天记录
        """
        return prompt

    def stream_answer(self, question: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        实现 BaseAgent 的 stream_answer 方法，先查 RAG 和网络和历史记录，再拼 prompt，只调用一次大模型，流式返回
        """
        rag = self._retrieve_knowledge(question)
        web = self._web_search(question)
        chat_history = kwargs.get("chat_history")
        prompt = self._build_prompt(question, rag, web, chat_history)
        # 假设 llm 支持 stream 方法
        return self.llm.stream(prompt)
