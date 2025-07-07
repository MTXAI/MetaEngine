import asyncio
import logging
import threading
import traceback
from typing import Callable, Tuple, Union, List

import queue
from queue import Queue


class PipelineCallback:
    def on_error(self, e: Exception):
        pass

    def on_stop(self, module: str=""):
        pass


class TODOPipelineCallback(PipelineCallback):
    def on_error(self, e: Exception):
        logging.info('error', e)

    def on_stop(self, module: str=""):
        logging.info('stop', module)


class Pipeline:
    def __init__(
            self,
            producer: Callable,
            consumer: Callable,
            generator: Callable=None,
            callbacks: Union[List[PipelineCallback], Tuple[PipelineCallback], PipelineCallback]=None,
            timeout: float=0.1,
            **kwargs
        ):
        self.name = kwargs.pop('name', 'Unknown')
        self.producer = producer
        self.consumer = consumer
        self.generator = generator
        if callbacks is not None:
            if isinstance(callbacks, PipelineCallback):
                self.callbacks = [callbacks]
            else:
                self.callbacks = list(callbacks)
        else:
            self.callbacks = callbacks

        self.queue = Queue()
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._produce_event = threading.Event()
        self._consume_event = threading.Event()

    def flush(self):
        self.queue.queue.clear()

    def produce_worker(self):
        try:
            for data in self.producer():
                if self._stop_event.is_set():
                    break
                self.queue.put(data)
        except Exception as e:
            for c in self.callbacks:
                c.on_error(e)
            traceback.print_exc()
        self._produce_event.set()
        for c in self.callbacks:
            c.on_stop(f"{self.name}.producer")

    def consume_worker(self):
        while not self._produce_event.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=self.timeout)
                self.consumer(data)
            except queue.Empty:
                continue
            except Exception as e:
                for c in self.callbacks:
                    c.on_error(e)
                traceback.print_exc()
        self._consume_event.set()
        for c in self.callbacks:
            c.on_stop(f"{self.name}.consumer")

    def shutdown(self):
        self._stop_event.set()
        for c in self.callbacks:
            c.on_stop(f"{self.name}")

class AsyncPipeline:
    def __init__(
            self,
            producer: Callable,
            consumer: Callable,
            generator: Callable=None,
            callbacks: Union[List[PipelineCallback], Tuple[PipelineCallback], PipelineCallback] = None,
            timeout: float=0.1,
            **kwargs
        ):
        self.name = kwargs.pop('name', 'Unknown')
        self.producer = producer
        self.consumer = consumer
        self.generator = generator
        if callbacks is not None:
            if isinstance(callbacks, PipelineCallback):
                self.callbacks = [callbacks]
            else:
                self.callbacks = list(callbacks)
        else:
            self.callbacks = []

        self.queue = asyncio.Queue()
        self.timeout = timeout
        self._stop_event = asyncio.Event()

    def flush(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()  # 如果使用了 join()，需要标记任务完成
            except asyncio.QueueEmpty:
                break

    async def produce_worker(self):
        try:
            async for data in self.producer():
                if self._stop_event.is_set():
                    break
                await self.queue.put(data)
        except Exception as e:
            for c in self.callbacks:
                c.on_error(e)
            traceback.print_exc()
        for c in self.callbacks:
            c.on_stop(f"{self.name}.producer")

    async def consume_worker(self):
        while not self._stop_event.is_set() or not self.queue.empty():
            try:
                data = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
                if asyncio.iscoroutinefunction(self.consumer):
                    await self.consumer(data)
                else:
                    self.consumer(data)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                for c in self.callbacks:
                    c.on_error(e)
                traceback.print_exc()
        for c in self.callbacks:
            c.on_stop(f"{self.name}.consumer")

    def shutdown(self):
        self._stop_event.set()
        for c in self.callbacks:
            c.on_stop(f"{self.name}")

