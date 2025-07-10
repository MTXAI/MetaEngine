import io
import logging
import traceback
from typing import Callable

import dashscope
import numpy as np
from dashscope.audio.tts_v2 import *

from engine.human.voice.voice import TTSModelWrapper
from engine.utils.sound import resample_sound_raw


class AliTTSCallback(ResultCallback):
    def __init__(self, on_data_fn: Callable):
        super().__init__()
        self.on_data_fn = on_data_fn

    def on_data(self, data: bytes) -> None:
        self.on_data_fn(data)


def _get_format(sample_rate: int = 16000, subtype: str = "PCM"):
    if subtype == "PCM":
        for f in [
            AudioFormat.PCM_8000HZ_MONO_16BIT,
            AudioFormat.PCM_16000HZ_MONO_16BIT,
            AudioFormat.PCM_22050HZ_MONO_16BIT,
            AudioFormat.PCM_24000HZ_MONO_16BIT,
            AudioFormat.PCM_44100HZ_MONO_16BIT,
            AudioFormat.PCM_48000HZ_MONO_16BIT,
        ]:
            if sample_rate == f.sample_rate:
                return f
            else:
                continue
        raise ValueError()
    elif subtype == "WAV":
        for f in [
            AudioFormat.WAV_8000HZ_MONO_16BIT,
            AudioFormat.WAV_16000HZ_MONO_16BIT,
            AudioFormat.WAV_22050HZ_MONO_16BIT,
            AudioFormat.WAV_24000HZ_MONO_16BIT,
            AudioFormat.WAV_44100HZ_MONO_16BIT,
            AudioFormat.WAV_48000HZ_MONO_16BIT,
        ]:
            if sample_rate == f.sample_rate:
                return f
            else:
                continue
        raise ValueError()
    else:
        raise ValueError()

class AliTTSWrapper(TTSModelWrapper):
    def __init__(
            self,
            model_str: str,
            api_key: str,
            voice_type: str,
            sample_rate: int,
    ):
        super().__init__()
        self.model_str = model_str
        self.api_key = api_key
        self.voice_type = voice_type
        self.sample_rate = sample_rate
        self.retry_count = 3

        dashscope.api_key = api_key
        self.audio_format = _get_format(self.sample_rate, "PCM")

        self.streaming_synthesizer = None
        self.synthesizer = None

    def _init_synthesizer(self):
        self.synthesizer = SpeechSynthesizer(
            model=self.model_str,
            voice=self.voice_type,
            format=self.audio_format,
        )

    def _on_data(self, data: bytes):
        speech = resample_sound_raw(data, self.sample_rate)
        return speech

    def _on_data_callback(self, fn: Callable):
        def on_data(data: bytes):
            speech = self._on_data(data)
            fn(speech)
        return on_data

    def _init_streaming_synthesizer(self, fn: Callable):
        callback = AliTTSCallback(self._on_data_callback(fn))
        self.streaming_synthesizer = SpeechSynthesizer(
            model=self.model_str,
            voice=self.voice_type,
            format=self.audio_format,
            callback=callback,
        )

    def reset(self, fn: Callable):
        self.inited = True
        self._init_streaming_synthesizer(fn)
        self._init_synthesizer()

    def complete(self):
        self.inited = False
        if self.streaming_synthesizer is not None:
            try:
                self.streaming_synthesizer.streaming_complete()
            except:
                pass
        if self.synthesizer is not None:
            try:
                self.synthesizer.streaming_complete()
            except:
                pass
        self.streaming_synthesizer = None
        self.synthesizer = None

    def streaming_inference(self, text: str):
        assert self.inited
        self.streaming_synthesizer.streaming_call(text)


    def inference(self, text):
        assert self.inited
        buffer_data = self.synthesizer.call(text)
        if buffer_data is None:
            return None
        return self._on_data(buffer_data)
