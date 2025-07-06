import asyncio
import io
import traceback

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

    async def _communicating(self, text):
        try:
            communicate = edge_tts.Communicate(text, self.voice_type)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    self.buffer.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            traceback.print_exc()
            raise e

    def streaming_inference(self, text):
        # return self.inference(text)
        raise NotImplementedError()

    def inference(self, text):
        asyncio.new_event_loop().run_until_complete(self._communicating(text))
        speech = resample_sound(self.buffer, self.sample_rate)
        self.buffer.seek(0)
        self.buffer.truncate()
        self.buffer.flush()
        return speech
