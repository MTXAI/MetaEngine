import io

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

        dashscope.api_key = api_key
        self.buffer = io.BytesIO()
        self.sample_rate = sample_rate

        callback = AliTTSCallback(self.buffer)
        self.synthesizer = SpeechSynthesizer(
            model=model_str,
            voice=voice_type,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=callback,
        )

    def streaming_inference(self, text):
        self.synthesizer.streaming_call(text)
        speech = resample_sound(self.buffer, self.sample_rate)
        self.buffer.seek(0)
        self.buffer.truncate()
        return speech

    def inference(self, text):
        self.buffer.write(self.synthesizer.call(text))
        speech = resample_sound(self.buffer, self.sample_rate)
        self.buffer.seek(0)
        self.buffer.truncate()
        return speech
