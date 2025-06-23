"""
由各种模型组装而来, 基于 AsyncPipeline 端到端运行, 如文本 -- 语音 -- 语音等

"""
import asyncio
from typing import List

from engine.utils.async_utils import AsyncPipeline, AsyncBridgeConsumer


class Player:
    pipelines: List[AsyncPipeline]
    def __init__(self):
        self.pipelines = []

    def add_pipeline(self, pipeline: AsyncPipeline, pipeline_bridge: AsyncBridgeConsumer=None):
        if len(self.pipelines) == 0:
            self.pipelines.append(pipeline)
        else:
            assert pipeline_bridge is not None
            self.pipelines[-1].add_consumer(pipeline_bridge)
            assert pipeline.producer is not None  # pipeline_bridge.to_producer()
            self.pipelines.append(pipeline)

    async def submit(self):
        tasks = []
        for p in self.pipelines:
            tasks.append(p.start())
        for p in self.pipelines:
            tasks.append(p.stop())
        await asyncio.gather(*tasks, return_exceptions=True)

    def run(self):
        asyncio.run(self.submit())


if __name__ == '__main__':
    from openai import AsyncOpenAI

    from engine.utils.async_utils import AsyncConsumerFactory, PipelineCallback

    client = AsyncOpenAI(
        # one api 生成的令牌
        api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
        base_url="http://localhost:3000/v1"
    )


    async def openai_producer():
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


    def openai_handler(content):
        # if len(content) > 2:
        #     raise Exception("<UNK>")
        return content


    def openai_handler_2(content):
        return content


    async def openai_consume_fn(data, processed_data=None):
        print(1, len(data), data)
        return data


    openai_consumer = AsyncConsumerFactory.with_consume_fn(openai_consume_fn, openai_handler)


    async def openai_consume_fn_2(data, processed_data=None):
        print(2, len(data), data)
        return data


    openai_consumer_2 = AsyncConsumerFactory.with_consume_fn(openai_consume_fn_2, openai_handler_2)

    stop_event = asyncio.Event()
    callback = PipelineCallback.with_events(stop_event=stop_event)
    pipeline_bridge = AsyncBridgeConsumer(stop_event=stop_event)
    pipeline = AsyncPipeline(
        openai_producer,
        openai_consumer,
        callback=callback)
    pipeline2 = AsyncPipeline(pipeline_bridge.to_producer(), consumers=[openai_consumer_2])

    player = Player()
    player.add_pipeline(pipeline)
    player.add_pipeline(pipeline2, pipeline_bridge)
    player.run()
