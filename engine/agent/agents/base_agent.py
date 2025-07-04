from abc import ABC, abstractmethod
from typing import Generator


class BaseAgent(ABC):
    @abstractmethod
    def stream_answer(self, question: str, **kwargs) -> Generator[str, None, None]:
        """
        接收问题，流式输出答案。
        :param question: 问题文本
        :param kwargs: 其他可选参数
        :return: 异步生成器，逐步yield答案片段
        """
        pass

    @abstractmethod
    def answer(self, question: str, **kwargs) -> str:
        pass
