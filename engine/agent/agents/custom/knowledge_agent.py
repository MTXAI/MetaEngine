from typing import Generator

from langchain_chroma import Chroma
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI

from engine.agent.agents.base_agent import BaseAgent


class KnowledgeAgent(BaseAgent):
    """
    只与大模型交互一次，先联网搜索和查找rag，将结果放入prompt，最后流式返回query结果
    """

    def __init__(self, model: ChatOpenAI, vector_db: Chroma):
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

    def _build_prompt(self, question: str, rag: str="", web: str="", chat_history: str="") -> str:
        rag_prompt = f"【内部知识库检索结果】\n{rag}" if rag != "" else ""
        web_prompt = f"【网络搜索结果】\n{web}" if web != "" else ""
        history_prompt = f"【历史聊天记录】\n{chat_history}" if chat_history != "" else ""
        prompt = f"""
            你是一个专业助手，请根据以下信息回答用户问题。
            {rag_prompt}
            {web_prompt}
            {history_prompt}
            【用户问题】
            {question}
            请结合上述所有信息，给出准确、简明的中文回答。注意优先级: 知识库检索结果 > 网络搜索结果 > 历史聊天记录, 当优先级较高的信息无法确认或不存在时, 参考低优先级的信息, 给出估计的答案
        """
        return prompt

    def stream_answer(self, question: str, **kwargs) -> Generator[str, None, None]:
        """
        实现 BaseAgent 的 stream_answer 方法，先查 RAG 和网络和历史记录，再拼 prompt，只调用一次大模型，流式返回
        """
        chat_history = kwargs.get("chat_history", "")
        use_web = kwargs.get("use_web")
        use_rag = kwargs.get("use_rag")
        rag = self._retrieve_knowledge(question) if use_rag else ""
        web = self._web_search(question) if use_web else ""
        prompt = self._build_prompt(question, rag, web, chat_history)
        # 假设 llm 支持 stream 方法
        for message_chunk in self.llm.stream(prompt):
            yield message_chunk.content

    def answer(self, question: str, **kwargs) -> str:
        chat_history = kwargs.get("chat_history", "")
        use_web = kwargs.get("use_web")
        use_rag = kwargs.get("use_rag")
        rag = self._retrieve_knowledge(question) if use_rag else ""
        web = self._web_search(question) if use_web else ""
        prompt = self._build_prompt(question, rag, web, chat_history)
        # 假设 llm 支持 stream 方法
        return self.llm.invoke(prompt).content
