import asyncio
from asyncio import AbstractEventLoop

import numpy as np
from typing import List, Optional

from engine.config import PlayerConfig
from engine.human.utils.data import Data


class AudioContainer:
    def __init__(
            self,
            config: PlayerConfig,
            targets: List[asyncio.Queue],
            loop: AbstractEventLoop,
            debug: bool = False,
    ):
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.frame_interval = float(self.sample_rate / self.fps)
        self.chunk_size = int(self.sample_rate/self.fps)
        self.targets = targets
        self._stop_event = asyncio.Event()
        self.loop = loop

        self.queue = asyncio.Queue()
        self.frame_fragment = None

        self.debug = debug
        self.i = 0

    async def consumer(self, data: Data):
        is_final = data.get("is_final")
        speech_data = data.get("data")
        if self.frame_fragment is not None:
            speech_data = np.concatenate([self.frame_fragment, speech_data])

        chunk_count = int((len(speech_data) - 1) / self.chunk_size) + 1
        for i in range(chunk_count):
            chunk = speech_data[i * self.chunk_size:(i + 1) * self.chunk_size]
            if not is_final and i == chunk_count - 1 and len(chunk) < self.chunk_size:
                self.frame_fragment = chunk
            else:
                await self.queue.put(chunk)
        return data

    async def send_audio_frame_worker(self):
        while not self._stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=self.frame_interval)
                state = 1
            except asyncio.TimeoutError:
                # 队列为空时发送空数据
                chunk = np.zeros(self.chunk_size, dtype=np.float32)
                state = 0

            for target in self.targets:
                await target.put(
                    Data(
                        data=chunk,
                        state=state,
                    )
                )

    async def shutdown(self):
        self._stop_event.set()
        await self.queue.join()


if __name__ == '__main__':
    from engine.utils.pipeline import AsyncPipeline
    from engine.config import WAV2LIP_PLAYER_CONFIG
    import time

    from engine.utils.pool import CoroutinePool
    async def run_async_pipeline():

        from os import PathLike
        import soundfile

        def soundfile_producer(f, chunk_size_or_fps):
            speech, sample_rate = soundfile.read(f)
            if isinstance(chunk_size_or_fps, int):
                chunk_stride = int(sample_rate / chunk_size_or_fps)  # sample rate 16000, fps 50, 20ms
            else:
                chunk_stride = chunk_size_or_fps[1] * 32  # [0, 10, 5] is 20ms
            async def produce_fn():
                total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
                for i in range(total_chunk_num):
                    speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
                    is_final = i == total_chunk_num - 1
                    yield Data(
                        data=speech_chunk,
                        final=is_final,
                    )

            return produce_fn

        s_f = '../../../tests/test_datas/asr.wav'

        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        audio_container = AudioContainer(
            WAV2LIP_PLAYER_CONFIG,
            [queue],
            asyncio.get_event_loop(),
            debug=True,
        )

        pipe = AsyncPipeline(
            producer=soundfile_producer(s_f, 10),
            consumer=audio_container.consumer,
        )

        async def read_data():
            i = 0
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                data: Data
                speech_data, state = data.get("data"), data.get("state")
                i += 1
                if state == 1:
                    print(f"Task read: {i}, {len(speech_data)}, {state}")

        async def shutdown_fn():
            await asyncio.sleep(10)
            pipe.shutdown()
            await audio_container.shutdown()
            stop_event.set()


        async with CoroutinePool() as pool:
            await pool.submit(pipe.produce_worker)
            await pool.submit(pipe.consume_worker)
            await pool.submit(read_data)
            await pool.submit(shutdown_fn)
            await pool.submit(audio_container.send_audio_frame_worker)

    asyncio.run(run_async_pipeline())

