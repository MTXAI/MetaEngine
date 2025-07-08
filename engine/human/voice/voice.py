from typing import Callable

import numpy as np
from torch import nn


class TTSModelWrapper(nn.Module):
    inited: bool = False
    def reset(self, fn: Callable):
        """
        主要用于流式 tts 中进行初始化
        :param fn: 用于接收流式 speech 数据, def fn(speech: np.ndarray) -> None
        :return:
        """
        self.inited = True

    def complete(self):
        """
        主要用于流式 tts 中结束并等待流式处理完成, 并重置
        :return:
        """
        self.inited = False

    def streaming_inference(self, text: str) -> None:
        """
        流式输入文本, 输出音频, 依赖 reset 和 complete
        """
        assert self.inited

    def inference(self, text: str) -> np.ndarray:
        """
        输入文本, 输出音频
        """
        assert self.inited
