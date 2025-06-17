## pipeline 文字 - 语音

import asyncio

import dashscope
import pyaudio
from dashscope.audio.tts_v2 import *

from engine.utils.async_utils import AsyncPipeline, PipelineCallback, AsyncConsumerFactory

dashscope.api_key = "sk-361f246a74c9421085d1d137038d5064"
model = "cosyvoice-v1"
voice = "longxiaochun"
class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        # print("websocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050, output=True
        )

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        # stop player
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        # print(f"recv speech synthsis message {message}")
        pass

    def on_data(self, data: bytes) -> None:
        # print("audio result length:", len(data))
        self._stream.write(data)
callback = Callback()
synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=callback,
    )

from openai import AsyncOpenAI
client = AsyncOpenAI(
            # one api 生成的令牌
            api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
            base_url="http://localhost:3000/v1"
        )

async def producer():
    chat_completion = await client.chat.completions.create(
        # model="doubao-1.5-lite-32k",
        model="qwen-turbo",
        messages=[
            {
                "role": "user",
                "content": "假设你正在带货一本英语单词书, 请简短准确且友好礼貌地回复弹幕问题: 怎么发货?, 只需回复问题, 无需问候",
            }
        ],
        stream=True,
    )
    async for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        yield content

def consume_fn(data, precessed_data=None):
    synthesizer.streaming_call(data)
    return data

class TTSCallback(PipelineCallback):
    def on_stop(self):
        super().on_stop()
        synthesizer.streaming_complete()

async def run_pipeline():
    pipeline = AsyncPipeline(
        producer=producer,
        consumers=[
            AsyncConsumerFactory.with_consume_fn(consume_fn)
        ],
        callback=TTSCallback(),
    )

    tasks = [
        pipeline.start(),
        pipeline.stop(),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(run_pipeline())

