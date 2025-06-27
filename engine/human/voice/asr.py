from os import PathLike
from typing import AsyncGenerator, Tuple

import soundfile

from engine.human.utils.data import SoundData
from engine.utils.pipeline import AsyncConsumer, AsyncConsumerFactory


## Producer
def soundfile_producer(f: PathLike, chunk_size: Tuple):
    async def produce_fn() -> AsyncGenerator[SoundData]:
        speech, sample_rate = soundfile.read(f)
        chunk_stride = chunk_size[1] * 960  # [0, 10, 5] is 600ms

        total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
        for i in range(total_chunk_num):
            speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            yield SoundData(
                sound=speech_chunk,
                stream=True,
                final=is_final,
            )
    return produce_fn


def sounddevice_producer():
    async def produce_fn() -> AsyncGenerator[SoundData]:
        pass


def websocket_producer():
    async def produce_fn() -> AsyncGenerator[SoundData]:
        pass


## Handler



## Consumer
def asr_consumer() -> AsyncConsumer:
    def consume_fn(data: SoundData, processed_data: SoundData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def async_asr_consumer() -> AsyncConsumer:
    async def consume_fn(data: SoundData, processed_data: SoundData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


## Callback

