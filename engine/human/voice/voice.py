from typing import Callable, Optional, Generator, Tuple

import numpy as np
from torch import nn

from engine.config import VoiceProcessorConfig
from engine.utils import Data


class VoiceProcessor:
    """
    todo 对于每一帧, 做各种后处理
    """
    def __init__(self, config: VoiceProcessorConfig):
        self.config = config

    def process(
            self,
            frame: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        return frame


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
        pass


class Voice:
    def __init__(
        self,
        tts_model: TTSModelWrapper,
        voice_processor: VoiceProcessor,
    ):
        self.tts_model = tts_model
        # todo process 做变声或其他处理
        self.voice_processor = voice_processor

    def speak(self, text: str) -> Optional[np.ndarray]:
        return self.tts_model.inference(text)

    def realtime_speak(self, generator: Generator[Tuple[str, bool], None, None], receiver: Callable) -> None:
        self.tts_model.reset(receiver)
        for text, is_final in generator:
            if is_final:
                break
            if len(text) > 0:
                self.tts_model.streaming_inference(text)
        self.tts_model.complete()

