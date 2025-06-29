import asyncio
import queue
import threading
from queue import Queue
from typing import Callable, Tuple, Union

import trio

from engine.human.utils.data import Data
from engine.utils.pool import TaskCallback, ThreadPool, TaskInfo


class PipelineCallback:
    def on_start(self):
        pass

    def on_error(self, e: Exception):
        pass

    def on_stop(self):
        pass


class Pipeline:
    def __init__(
            self,
            producer: Callable,
            consumer: Callable,
            generator: Callable=None,
            timeout: float=0.1,
            **kwargs
        ):
        self.producer = producer
        self.consumer = consumer
        self.generator = generator
        self.out_queue = Queue()
        self.queue = Queue()
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._produce_event = threading.Event()
        self._consume_event = threading.Event()

    def produce_worker(self):
        for data in self.producer():
            if self._stop_event.is_set():
                break
            self.queue.put(data)
        self._produce_event.set()

    def consume_worker(self):
        while not self._produce_event.is_set() or not self.queue.empty():
            try:
                data = self.queue.get(timeout=self.timeout)
            except queue.Empty:
                continue
            data = self.consumer(data)
            self.out_queue.put(data)
        self._consume_event.set()

    def wait_for_results(self):
        while not self._consume_event.is_set() or not self.out_queue.empty():
            try:
                data = self.out_queue.get(timeout=self.timeout)
                if self.generator is not None:
                    for _data in self.generator(data):
                        yield _data
                else:
                    yield data
            except queue.Empty:
                continue

    def shutdown(self):
        self._stop_event.set()



class AsyncPipeline:
    def __init__(
            self,
            producer: Callable,
            consumer: Callable,
            generator: Callable=None,
            timeout: float=0.1,
            channel_size: int = 100,
            **kwargs
        ):
        self.producer = producer
        self.consumer = consumer
        self.generator = generator
        self.sender, self.receiver = trio.open_memory_channel(channel_size)
        self.output_sender, self.output_receiver = trio.open_memory_channel(channel_size)
        self.timeout = timeout
        self._stop_event = asyncio.Event()

    async def produce_worker(self):
        async for data in self.producer():
            if self._stop_event.is_set():
                break
            await self.sender.send(data)
        await self.sender.aclose()

    async def consume_worker(self):
        async for data in self.receiver:
            await self.output_sender.send(data)
        await self.receiver.aclose()
        await self.output_sender.aclose()

    async def wait_for_results(self):
        async for data in self.output_receiver:
            if self.generator is not None:
                for _data in self.generator(data):
                    yield _data
            else:
                yield data
        await self.output_receiver.aclose()

    def shutdown(self):
        self._stop_event.set()

if __name__ == '__main__':
    class SimpleCallback(TaskCallback):
        def on_submit(self, future, task_info):
            print(f"Task submitted: {future}, info: {task_info}")

        def on_schedule(self, future, task_info):
            print(f"Task scheduled: {future}, info: {task_info}")

        def on_complete(self, future, task_info):
            try:
                result = future.result()
                print(f"Task completed, Result: {result}, info: {task_info}")
            except Exception as e:
                print(f"Task completed, Error: {e}, info: {task_info}")

    def run_pipeline():

        from os import PathLike
        import soundfile

        total_chunk_num = 0

        def soundfile_producer(f: Union[PathLike, str], chunk_size_or_fps: Union[Tuple, int]):
            speech, sample_rate = soundfile.read(f)
            if isinstance(chunk_size_or_fps, int):
                chunk_stride = int(sample_rate / chunk_size_or_fps)  # sample rate 16000, fps 50, 20ms
            else:
                chunk_stride = chunk_size_or_fps[1] * 32  # [0, 10, 5] is 20ms

            def produce_fn():
                global total_chunk_num
                total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
                for i in range(total_chunk_num):
                    speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
                    is_final = i == total_chunk_num - 1
                    yield Data(
                        data=i,
                        final=is_final,
                    )

            return produce_fn

        s_f = '../../tests/test_datas/asr.wav'

        with ThreadPool(max_workers=4, max_queue_size=5) as pool:
            def consume_fn(data):
                return data.data

            pipe = Pipeline(
                producer=soundfile_producer(s_f, 50),
                consumer=consume_fn,
            )
            try:
                submitted_futures = []
                future = pool.submit(
                    pipe.produce_worker,
                    task_info=TaskInfo(
                        name="Task producer",
                    ),
                    callback=SimpleCallback(),
                )
                submitted_futures.append(future)

                future = pool.submit(
                    pipe.consume_worker,
                    task_info=TaskInfo(
                        name="Task consumer",
                    ),
                    callback=SimpleCallback(),
                )
                submitted_futures.append(future)
            except Exception as e:
                print(e)

            import time
            def shutdown_fn():
                time.sleep(3)
                pipe.shutdown()

            def read_data():
                i = 0
                for data in pipe.wait_for_results():
                    print(f"Task read: {i} -- {data}")
                    i += 1

            pool.submit(
                read_data,
                task_info=TaskInfo(
                    name="read",
                ),
                callback=SimpleCallback(),
            )

            pool.submit(
                shutdown_fn,
                task_info=TaskInfo(
                    name="shutdown",
                ),
                callback=SimpleCallback(),
            )

            pool.shutdown(wait=True)

    async def run_async_pipeline():

        import trio
        from os import PathLike
        import soundfile

        total_chunk_num = 0

        def soundfile_producer(f: Union[PathLike, str], chunk_size_or_fps: Union[Tuple, int]):
            speech, sample_rate = soundfile.read(f)
            if isinstance(chunk_size_or_fps, int):
                chunk_stride = int(sample_rate / chunk_size_or_fps)  # sample rate 16000, fps 50, 20ms
            else:
                chunk_stride = chunk_size_or_fps[1] * 32  # [0, 10, 5] is 20ms

            async def produce_fn():
                global total_chunk_num
                total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
                for i in range(total_chunk_num):
                    speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
                    is_final = i == total_chunk_num - 1
                    yield Data(
                        data=i,
                        final=is_final,
                    )

            return produce_fn

        s_f = '../../tests/test_datas/asr.wav'

        async def consume_fn(data):
            return data.data

        pipe = AsyncPipeline(
            producer=soundfile_producer(s_f, 50),
            consumer=consume_fn,
        )

        async def read_data():
            i = 0
            async for data in pipe.wait_for_results():
                if data is None:
                    return
                print(f"Task read: {i} -- {data}")
                i += 1

        async def shutdown_fn():
            await asyncio.sleep(3)
            pipe.shutdown()

        async with trio.open_nursery() as nursery:
            nursery.start_soon(pipe.produce_worker)
            nursery.start_soon(pipe.consume_worker)
            nursery.start_soon(read_data)

    # trio.run(run_async_pipeline)
    run_pipeline()
    # trio.run(run_async_pipeline)

