import asyncio
import logging
from typing import Any, Callable, List, Tuple, Optional, Union


class PipelineCallback:
    def on_start(self):
        pass

    def on_error(self, e: Exception):
        pass

    def on_stop(self):
        pass

    @staticmethod
    def with_events(
        start_event: asyncio.Event = None,
        error_event: asyncio.Event = None,
        stop_event: asyncio.Event = None,
    ):
        class _PipelineCallBackWithEvent(PipelineCallback):
            def __init__(
                    self,
                    start_event: asyncio.Event = None,
                    error_event: asyncio.Event = None,
                    stop_event: asyncio.Event = None,
            ):
                super().__init__()
                self.start_event = start_event
                self.error_event = error_event
                self.stop_event = stop_event

            def on_start(self):
                if self.start_event:
                    self.start_event.set()

            def on_error(self, e: Exception):
                if self.error_event:
                    self.error_event.set()

            def on_stop(self):
                if self.stop_event:
                    self.stop_event.set()

        return _PipelineCallBackWithEvent(
            start_event=start_event,
            error_event=error_event,
            stop_event=stop_event,
        )


class AsyncConsumer:
    def __init__(self, handler: Callable=None):
        self.handler = handler

    def has_handler(self):
        return self.handler is not None

    async def consume(self, data: Any, processed_data: Any=None, **kwargs) -> Any:
        """
        async consume data maybe processed
        """
        pass

    def on_consume(self, data: Any, **kwargs) -> Any:
        """
        process data before consume
        """
        if self.has_handler():
            return self.handler(data, **kwargs)
        return data


class AsyncBridgeConsumer(AsyncConsumer):
    """
    wrap consumer as producer
    """
    def __init__(self, stop_event, generator=None, timeout: float=0.1):
        super().__init__()
        self.stop_event = stop_event
        self.generator = generator
        self.queue = asyncio.Queue()
        self.timeout = timeout

    async def consume(self, data: Any, processed_data: Any=None, **kwargs) -> Any:
        await self.queue.put(data)
        return data

    def to_producer(self):
        async def producer():
            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    data = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
                    if self.generator is not None:
                        async for _data in self.generator(data):
                            yield _data
                    else:
                        yield data
                except asyncio.TimeoutError:
                    continue
        return producer


class AsyncConsumerFactory:
    @staticmethod
    def with_consume_fn(consume_fn, handler=None):
        class _AsyncConsumer(AsyncConsumer):
            def __init__(self):
                super().__init__(handler=handler)

            async def consume(self, data: Any, processed_data: Any = None, **kwargs) -> Any:
                if asyncio.iscoroutinefunction(consume_fn):
                    data = await consume_fn(data, processed_data, **kwargs)
                else:
                    data = consume_fn(data, **kwargs)
                return data

        return _AsyncConsumer()


class AsyncPipeline:
    """
    任何错误都会终止任务, 且不可恢复, 不适合长时间运行, 需要避免单个 pipeline 过长, 或分叉太多
    """
    def __init__(
            self,
            producer: Callable=None,
            consumers: Union[List[AsyncConsumer], Tuple[AsyncConsumer], AsyncConsumer]=None,
            callbacks: Union[List[PipelineCallback], Tuple[PipelineCallback], PipelineCallback]=None,
            timeout: float=0.1,
            **kwargs
        ):
        self.name = kwargs.get("name", "Unknown")
        self.producer = producer

        if consumers is None:
            consumers = []
        if not hasattr(consumers, "__len__"):
            consumers = [consumers]
        self.consumers: List[AsyncConsumer] = list(consumers)

        if callbacks is None:
            callbacks = []
        if not hasattr(callbacks, "__len__"):
            callbacks = [callbacks]
        self.callbacks: List[PipelineCallback] = list(callbacks)

        self.queues = []
        for i in range(len(self.consumers)):
            self.queues.append(asyncio.Queue())
        self.input_queue = self.queues[0] if len(self.queues) > 0 else None
        self.error_event = asyncio.Event()
        self.done_event = asyncio.Event()

        self.running = False
        self.daemon_task = None
        self.producer_task = None
        self.consumer_tasks = []

        ## kwargs
        self.timeout = timeout

    def set_producer(self, producer):
        self.producer = producer

    def add_consumer(self, consumer: AsyncConsumer):
        self.consumers.append(consumer)
        self.queues.append(asyncio.Queue())
        if len(self.queues) == 1:
            self.input_queue = self.queues[0]

    def add_callback(self, callback: PipelineCallback):
        self.callbacks.append(callback)

    async def __daemon(self):
        if self.error_event is None:
            return
        while self.running:
            try:
                if self.error_event.is_set():
                    self.done_event.set()
                    logging.error(f'pipeline [{self.name}] quit...')
            finally:
                await asyncio.sleep(0.1)

    async def __produce_worker(self) -> None:
        try:
           async for data in self.producer():
                if self.running:
                    await self.input_queue.put(data)
        except Exception as e:
            logging.error(f'pipeline [{self.name}] detect a error [{e}]')
            self.error_event.set()
            for c in self.callbacks:
                c.on_error(e)
        finally:
            self.done_event.set()

    async def __consume_worker(self, idx) -> None:
        while self.running:
            try:
                data = await asyncio.wait_for(self.queues[idx].get(), timeout=self.timeout)
                if self.consumers[idx].has_handler():
                    processed_data = self.consumers[idx].on_consume(data)
                    new_data = await self.consumers[idx].consume(data, processed_data)
                else:
                    new_data = await self.consumers[idx].consume(data)
                self.queues[idx].task_done()
                if idx < len(self.consumers)-1:
                    await self.queues[idx+1].put(new_data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f'pipeline [{self.name}] detect a error [{e}]')
                self.error_event.set()
                for c in self.callbacks:
                    c.on_error(e)

    async def start(self) -> None:
        assert self.producer is not None
        assert len(self.consumers) > 0
        if not self.running:
            self.running = True
            self.daemon_task = asyncio.ensure_future(self.__daemon())
            self.producer_task = asyncio.ensure_future(self.__produce_worker())
            for i, h in enumerate(self.consumers):
                self.consumer_tasks.append(asyncio.create_task(self.__consume_worker(i)))
            for c in self.callbacks:
                c.on_start()

    async def stop(self) -> None:
        if self.running:
            if not self.done_event.is_set():
                await self.done_event.wait()

            if not self.error_event or not self.error_event.is_set():
                for q in self.queues:
                    await q.join()

            self.running = False
            tasks = [self.daemon_task, self.producer_task]
            tasks.extend(self.consumer_tasks)
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for c in self.callbacks:
                c.on_stop()


class AsyncPipelineRunner:
    pipelines: List[AsyncPipeline]
    def __init__(self):
        self.pipelines = []
        self.stop_event = asyncio.Event()

    def add_pipeline(self, pipeline: AsyncPipeline):
        """
        Concat multiple pipelines with bridges
        """
        if len(self.pipelines) == 0:
            assert pipeline.producer is not None
            self.pipelines.append(pipeline)
        else:
            # new bridge and callback
            pipeline_bridge = AsyncBridgeConsumer(stop_event=self.stop_event)
            callback = PipelineCallback.with_events(stop_event=self.stop_event)

            # add consumer and callback
            self.pipelines[-1].add_consumer(pipeline_bridge)
            self.pipelines[-1].add_callback(callback)

            # set producer
            if pipeline.producer is not None:
                logging.warning(f"pipeline already has a producer, will be replaced by bridge")
            pipeline.set_producer(pipeline_bridge.to_producer())
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

    client = AsyncOpenAI(
        # one api 生成的令牌
        api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
        base_url="http://localhost:3000/v1"
    )


    def openai_producer():
        async def produce_fn():
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
        return produce_fn


    def openai_handler(content):
        return content


    def openai_consumer():
        handler = openai_handler
        async def consume_fn(data, processed_data=None):
            print(len(data), data)
            return data
        return AsyncConsumerFactory.with_consume_fn(consume_fn, handler)


    import dashscope
    import pyaudio
    from dashscope.audio.tts_v2 import *

    class _Callback(ResultCallback):
        _player = None
        _stream = None

        def on_open(self):
            # print("websocket is open.")
            self._player = pyaudio.PyAudio()
            self._stream = self._player.open(
                format=pyaudio.paInt16, channels=1, rate=22050, output=True
            )

        def on_complete(self):
            logging.info("speech synthesis task complete successfully.")

        def on_error(self, message: str):
            logging.info(f"speech synthesis task failed, {message}")

        def on_close(self):
            logging.info("websocket is closed.")
            # stop player
            self._stream.stop_stream()
            self._stream.close()
            self._player.terminate()

        def on_event(self, message):
            logging.info(f"recv speech synthsis message {message}")
            pass

        def on_data(self, data: bytes) -> None:
            logging.info("audio result length:", len(data))
            self._stream.write(data)

    dashscope.api_key = "sk-361f246a74c9421085d1d137038d5064"
    model = "cosyvoice-v1"
    voice = "longxiaochun"
    callback = _Callback()
    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=callback,
    )

    def openai_consumer_2():
        handler = None
        async def consume_fn(data, processed_data=None):
            synthesizer.streaming_call(data)
            return data
        return AsyncConsumerFactory.with_consume_fn(consume_fn, handler)

    class TTSCallback(PipelineCallback):
        def on_stop(self):
            synthesizer.streaming_complete()

    runner = AsyncPipelineRunner()

    pipeline = AsyncPipeline(
        openai_producer(),
        # consumers=openai_consumer(),
    )
    runner.add_pipeline(pipeline)

    pipeline2 = AsyncPipeline(
        consumers=openai_consumer_2(),
        callbacks=TTSCallback(),
    )
    runner.add_pipeline(pipeline2)

    runner.run()

