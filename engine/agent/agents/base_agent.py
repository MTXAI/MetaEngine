from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseAgent(ABC):
    @abstractmethod
    async def stream_answer(self, question: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        接收问题，流式输出答案。
        :param question: 问题文本
        :param kwargs: 其他可选参数
        :return: 异步生成器，逐步yield答案片段
        """
        pass