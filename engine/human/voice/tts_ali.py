import io
import logging
import traceback
from typing import Callable

import dashscope
import numpy as np
from dashscope.audio.tts_v2 import *

from engine.human.voice.voice import TTSModelWrapper
from engine.utils.sound import resample_sound


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
        try:
            speech = resample_sound(data, self.sample_rate)
            return speech
        except Exception as e:
            logging.info(f"Failed to resample audio, error: {e}")
            traceback.print_exc()
            return None

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
        try:
            self.streaming_synthesizer.streaming_call(text)
        except TimeoutError:
            logging.info(f"Timeout error")
            return None
        except Exception as e:
            logging.info(f"Failed to stream inference: {e}")
            traceback.print_exc()
            return None

    def inference(self, text):
        assert self.inited
        try:
            buffer_data = self.synthesizer.call(text)
            if buffer_data is None:
                return None
            return self._on_data(buffer_data)
        except TimeoutError:
            return None
        except Exception as e:
            logging.info(f"Failed to inference: {e}")
            traceback.print_exc()
            return None


if __name__ == '__main__':
    # coding=utf-8
    # Installation instructions for pyaudio:
    # APPLE Mac OS X
    #   brew install portaudio
    #   pip install pyaudio
    # Debian/Ubuntu
    #   sudo apt-get install python-pyaudio python3-pyaudio
    #   or
    #   pip install pyaudio
    # CentOS
    #   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
    # Microsoft Windows
    #   python -m pip install pyaudio

    import pyaudio
    import dashscope
    from dashscope.audio.tts_v2 import *

    from http import HTTPStatus
    from dashscope import Generation

    from engine.utils.data import Data

    # 若没有将API Key配置到环境变量中，需将下面这行代码注释放开，并将apiKey替换为自己的API Key
    dashscope.api_key = "sk-361f246a74c9421085d1d137038d5064"
    model = "cosyvoice-v1"
    voice = "longxiaochun"

    tts_model = AliTTSWrapper(
        model_str="cosyvoice-v1",
        api_key="sk-361f246a74c9421085d1d137038d5064",
        voice_type="longxiaochun",
        sample_rate=16000,
    )


    def fn(speech: np.ndarray):
        if speech is None:
            logging.info("speech is None")
            return
        audio_data = Data(
            data=speech,
            is_final=False,
        )
        logging.info(len(audio_data.data))

    def synthesizer_with_llm():
        tts_model.reset(fn)

        messages = [{"role": "user", "content": "请介绍一下你自己"}]
        responses = Generation.call(
            model="qwen-turbo",
            messages=messages,
            result_format="message",  # set result format as 'message'
            stream=True,  # enable stream output
            incremental_output=True,  # enable incremental output
        )
        for response in responses:
            if response.status_code == HTTPStatus.OK:
                logging.info(response.output.choices[0]["message"]["content"], end="")
                tts_model.streaming_inference(response.output.choices[0]["message"]["content"])
            else:
                logging.info(
                    "Request id: %s, Status code: %s, error code: %s, error message: %s"
                    % (
                        response.request_id,
                        response.status_code,
                        response.code,
                        response.message,
                    )
                )
        tts_model.complete()


    synthesizer_with_llm()
