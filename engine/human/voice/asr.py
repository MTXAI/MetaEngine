from os import PathLike
from typing import AsyncGenerator, Tuple, Union

import soundfile

from engine.human.utils.data import Data


## Producer
def soundfile_producer(f: Union[PathLike, str], chunk_size_or_fps: Union[Tuple, int]):
    speech, sample_rate = soundfile.read(f)
    if isinstance(chunk_size_or_fps, int):
        chunk_stride = int(sample_rate / chunk_size_or_fps)  # sample rate 16000, fps 50, 20ms
    else:
        chunk_stride = chunk_size_or_fps[1] * 32  # [0, 10, 5] is 20ms
    def produce_fn():
        total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
        for i in range(total_chunk_num):
            speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
            is_final = i == total_chunk_num - 1
            yield Data(
                data=speech_chunk,
                final=is_final,
            )
    return produce_fn


def sounddevice_producer():
    async def produce_fn() -> AsyncGenerator:
        pass


def websocket_producer():
    async def produce_fn() -> AsyncGenerator:
        pass


## Handler



## Consumer
# def asr_consumer() -> AsyncConsumer:
#     def consume_fn(data: Data, processed_data: Data=None):
#         pass
#
#     handler = None
#     return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)
#
#
# def async_asr_consumer() -> AsyncConsumer:
#     async def consume_fn(data: Data, processed_data: Data=None):
#         pass
#
#     handler = None
#     return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


## Callback

