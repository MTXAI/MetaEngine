
## pipeline 文字 - 语音 done 一个 pipeline
## pipeline 语音 - 文字 done 一个 pipeline
## pipeline 语音 - 文字 - 文字 done 一个 pipeline
## pipeline 语音 - 文字 + 文字 - 语音 done, 两个 pipeline
## pipeline 文字 - 文字 - 语音+视频 - 视频 两个 pipeline + 一个 player 匹配合成语音与视频
## pipeline 语音 - 文字 - 文字 - 语音+视频 - 视频 三个 pipeline + 一个 player 匹配合成语音与视频

from os import PathLike
from typing import AsyncGenerator, Tuple

import soundfile

from engine.agent.agents.smol.agents import QaAgent
from engine.human.utils.data import Data
from engine.utils.pipeline import AsyncConsumer, AsyncConsumerFactory, PipelineCallback


## Producer
def completion_producer(agent_client: QaAgent, prompt: str):
    async def produce_fn() -> AsyncGenerator:
        pass
        # chat_completion = await agent_client.chat.completions.create(
        #     # model="doubao-1.5-lite-32k",
        #     model="qwen-turbo",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": "假设你正在带货一本英语单词书, 请简短准确且友好礼貌地回复弹幕问题: 怎么发货?, 只需回复问题, 无需问候",
        #         }
        #     ],
        #     stream=True,
        # )
        # async for chunk in chat_completion:
        #     content = chunk.choices[0].delta.content
        #     yield TextData(
        #         text=content,
        #         stream=True,
        #         final=False,
        #     )


## Handler
def text_fileter_handler(data: Data):
    return data


## Consumer
def tts_consumer() -> AsyncConsumer:
    handler = None

    def consume_fn(data: Data, processed_data: Data=None):
        return data

    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def async_tts_consumer() -> AsyncConsumer:
    async def consume_fn(data: Data, processed_data: Data=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


## Callback
class FinalizerCallback(PipelineCallback):
    def __init__(self, client):
        self.client = client

    def on_stop(self):
        super().on_stop()
        self.client.finalizer()  # just an example

