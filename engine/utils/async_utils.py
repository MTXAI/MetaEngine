import asyncio
from typing import Any, Callable, List


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
    任何错误都可能会导致阻塞, 且不可恢复, 因此不适合长时间运行
    """
    def __init__(
            self,
            producer: Callable,
            consumers: List[AsyncConsumer],
            callback: PipelineCallback=None,
            timeout: float=0.1,
        ):
        self.producer = producer
        self.consumers: List[AsyncConsumer] = consumers
        self.callback = callback if callback is not None else PipelineCallback()

        self.queues = []
        for i in range(len(consumers)):
            self.queues.append(asyncio.Queue())
        self.input_queue = self.queues[0]
        self.error_event = asyncio.Event()
        self.done_event = asyncio.Event()

        self.running = False
        self.daemon_task = None
        self.producer_task = None
        self.consumer_tasks = []

        ## kwargs
        self.timeout = timeout

    async def __daemon(self):
        if self.error_event is None:
            return
        while self.running:
            try:
                if self.error_event.is_set():
                    self.done_event.set()
                    print(f'detect a error, quit...')
            finally:
                await asyncio.sleep(0.1)

    async def __produce_worker(self) -> None:
        try:
           async for data in self.producer():
                if self.running:
                    await self.input_queue.put(data)
        except Exception as e:
            self.error_event.set()
            self.callback.on_error(e)
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
                self.error_event.set()
                self.callback.on_error(e)

    async def start(self) -> None:
        if not self.running:
            self.running = True
            self.daemon_task = asyncio.ensure_future(self.__daemon())
            self.producer_task = asyncio.ensure_future(self.__produce_worker())
            for i, h in enumerate(self.consumers):
                self.consumer_tasks.append(asyncio.create_task(self.__consume_worker(i)))
            self.callback.on_start()

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
            self.callback.on_stop()


if __name__ == '__main__':

    async def main():
        from openai import AsyncOpenAI
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
        pipeline = AsyncPipeline(openai_producer, [openai_consumer, pipeline_bridge], callback=callback)
        pipeline2 = AsyncPipeline(pipeline_bridge.to_producer(), [openai_consumer_2])


        tasks = [
            pipeline.start(),
            pipeline2.start(),
            pipeline.stop(),
            pipeline2.stop(),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(main())

