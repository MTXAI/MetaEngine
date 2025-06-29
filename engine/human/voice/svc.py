
## pipeline 文字 - 语音 done 一个 pipeline
## pipeline 语音 - 文字 done 一个 pipeline
## pipeline 语音 - 文字 - 文字 done 一个 pipeline
## pipeline 语音 - 文字 + 文字 - 语音 done, 两个 pipeline
## pipeline 文字 - 文字 - 语音+视频 - 视频 两个 pipeline + 一个 player 匹配合成语音与视频
## pipeline 语音 - 文字 - 文字 - 语音+视频 - 视频 三个 pipeline + 一个 player 匹配合成语音与视频

from engine.human.utils.data import Data
from engine.utils.pipeline import AsyncConsumer, AsyncConsumerFactory


## Producer



## Handler



## Consumer
def svc_consumer() -> AsyncConsumer:
    def consume_fn(data: Data, processed_data: Data=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def async_svc_consumer() -> AsyncConsumer:
    async def consume_fn(data: Data, processed_data: Data=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


## Callback


