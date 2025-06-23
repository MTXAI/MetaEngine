from os import PathLike
from typing import Any, AsyncGenerator, Tuple

import soundfile
import sounddevice

from engine.agent.agents.smol_agent import SmolAgent
from engine.human.player.data import AudioData, TextData


async def completion_producer(agent_client: SmolAgent, prompt: str) -> AsyncGenerator[TextData]:
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


async def soundfile_producer(f: PathLike, chunk_size: Tuple) -> AsyncGenerator[AudioData]:
    speech, sample_rate = soundfile.read(f)
    chunk_stride = chunk_size[1] * 960  # [0, 10, 5] is 600ms

    total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        yield AudioData(
            speech=speech_chunk,
            stream=True,
            final=is_final,
        )


async def sounddevice_producer() -> AsyncGenerator[AudioData]:
    pass



