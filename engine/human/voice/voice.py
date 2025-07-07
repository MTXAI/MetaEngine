from typing import Callable

import numpy as np
from torch import nn


class TTSModelWrapper(nn.Module):
    inited: bool = False
    def reset(self, fn: Callable):
        self.inited = True

    def complete(self):
        self.inited = False

    def streaming_inference(self, text: str) -> None:
        """
        流式输入文本, 输出音频
        """
        assert self.inited

    def inference(self, text: str) -> np.ndarray:
        """
        输入文本, 输出音频
        """
        assert self.inited
