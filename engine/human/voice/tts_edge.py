import asyncio
import io
import traceback
from typing import Callable

import edge_tts

from engine.human.voice.voice import TTSModelWrapper
from engine.utils.sound import resample_sound


class EdgeTTSWrapper(TTSModelWrapper):
    def __init__(
            self,
            voice_type: str,
            sample_rate: int,
    ):
        super().__init__()

        self.buffer = io.BytesIO()
        self.voice_type = voice_type
        self.sample_rate = sample_rate

    def _communicating(self, text):
        try:
            communicate = edge_tts.Communicate(text, self.voice_type)
            for chunk in communicate.stream_sync():
                if chunk["type"] == "audio":
                    self.buffer.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            print(f"Failed to communicate: {e}")
            traceback.print_exc()
            return

    def reset(self, fn: Callable):
        self.inited = True

    def complete(self):
        self.inited = False

    def streaming_inference(self, text):
        assert self.inited
        # todo streaming 支持
        return self.inference(text)

    def inference(self, text):
        assert self.inited
        self._communicating(text)
        speech = resample_sound(self.buffer, self.sample_rate)
        self.buffer.seek(0)
        self.buffer.truncate()
        self.buffer.flush()
        return speech
