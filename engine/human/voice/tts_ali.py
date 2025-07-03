import asyncio
import io
import traceback

import dashscope
import numpy as np
import soundfile
from dashscope.audio.tts_v2 import *

from engine.utils.sound import resample_sound
from engine.human.voice.voice import TTSModelWrapper


class AliTTSCallback(ResultCallback):
    def __init__(self, buffer):
        self.buffer = buffer

    def on_data(self, data: bytes) -> None:
        self.buffer.write(data)


class AliTTSWrapper(TTSModelWrapper):
    def __init__(
            self,
            model_str: str,
            api_key: str,
            voice_type: str,
            sample_rate: int,
    ):
        super().__init__()
        self.retry_count = 3

        dashscope.api_key = api_key
        self.buffer = io.BytesIO()
        self.sample_rate = sample_rate

        audio_format = AudioFormat.WAV_22050HZ_MONO_16BIT
        self.synthesizer = SpeechSynthesizer(
            model=model_str,
            voice=voice_type,
            format=audio_format,
        )
        callback = AliTTSCallback(self.buffer)
        self.streaming_synthesizer = SpeechSynthesizer(
            model=model_str,
            voice=voice_type,
            format=audio_format,
            callback=callback,
        )

    # todo @zjh, retry

    async def _streaming_call(self, text):
        for _ in range(self.retry_count):
            try:
                self.streaming_synthesizer.streaming_call(text)
                return
            except TimeoutError:
                continue
            except Exception as e:
                traceback.print_exc()
                raise e

    def streaming_inference(self, text):
        asyncio.new_event_loop().run_until_complete(self._streaming_call(text))
        speech = resample_sound(self.buffer, self.sample_rate)
        self.buffer.seek(0)
        self.buffer.truncate()
        self.buffer.flush()
        return speech

    def inference(self, text):
        for _ in range(self.retry_count):
            try:
                buffer_data = self.synthesizer.call(text)
                if buffer_data is None:
                    continue
                self.buffer.write(buffer_data)
                speech = resample_sound(self.buffer, self.sample_rate)
                self.buffer.seek(0)
                self.buffer.truncate()
                self.buffer.flush()
                return speech
            except TimeoutError:
                continue
            except Exception as e:
                traceback.print_exc()
                raise e
        return None

