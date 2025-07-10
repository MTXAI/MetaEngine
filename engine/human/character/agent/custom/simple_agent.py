from typing import Generator

from langchain_openai import ChatOpenAI

from engine.human.character.agent.base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    """
    只与大模型交互一次，先联网搜索和查找rag，将结果放入prompt，最后流式返回query结果
    """

    def __init__(self, model: ChatOpenAI):
        self.llm = model

    def _build_prompt(self, question: str) -> str:
        prompt = f"""
            你是一个专业助手，请根据以下信息回答用户问题。
            
            【用户问题】
            {question}
            
            给出准确、简明的中文回答。
        """
        return prompt

    def stream_answer(self, question: str, **kwargs) -> Generator[str, None, None]:
        prompt = self._build_prompt(question)
        for message_chunk in self.llm.stream(prompt):
            yield message_chunk.content

    def answer(self, question: str, **kwargs) -> str:
        prompt = self._build_prompt(question)
        return self.llm.invoke(prompt).content
