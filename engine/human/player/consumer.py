from engine.human.player.data import TextData, AudioData
from engine.utils.async_utils import AsyncConsumerFactory, AsyncConsumer


def tts_consumer() -> AsyncConsumer:
    def consume_fn(data: TextData, processed_data: TextData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def async_tts_consumer() -> AsyncConsumer:
    async def consume_fn(data: TextData, processed_data: TextData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def asr_consumer() -> AsyncConsumer:
    def consume_fn(data: AudioData, processed_data: AudioData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def async_asr_consumer() -> AsyncConsumer:
    async def consume_fn(data: AudioData, processed_data: AudioData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def svc_consumer() -> AsyncConsumer:
    def consume_fn(data: AudioData, processed_data: AudioData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)


def async_svc_consumer() -> AsyncConsumer:
    async def consume_fn(data: AudioData, processed_data: AudioData=None):
        pass

    handler = None
    return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)
