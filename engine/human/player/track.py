import asyncio
import fractions
import time
from typing import Union

from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame
from av.frame import Frame
from av.packet import Packet

from engine.config import PlayerConfig


class StreamTrackSync:
    def __init__(self, config: PlayerConfig):
        self.fps = config.fps

        self.audio_queue = asyncio.Queue(self.fps*config.frame_multiple+self.fps)
        self.video_queue = asyncio.Queue(self.fps+self.fps)

        self.audio_ptime = config.audio_ptime
        self.video_ptime = config.video_ptime
        self.sample_rate = config.sample_rate
        self.clock_rate = config.clock_rate
        self.audio_timebase = fractions.Fraction(1, self.sample_rate)
        self.video_timebase = fractions.Fraction(1, self.clock_rate)

        self.audio_timestamp = 0
        self.video_timestamp = 0
        self.audio_frame_count = 0
        self.video_frame_count = 0
        self.start_time = 0.0

        self.consumed_frame_count = 0

    def _clear_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()  # 如果使用了 join()，需要标记任务完成
            except asyncio.QueueEmpty:
                break

    def flush(self):
        self._clear_queue(self.audio_queue)
        self._clear_queue(self.video_queue)
        return self.consumed_frame_count

    async def put_audio_frame(self, frame: AudioFrame):
        await self.audio_queue.put(frame)

    async def put_video_frame(self, frame: VideoFrame):
        await self.video_queue.put(frame)

    async def get_audio_frame(self) -> AudioFrame:
        frame = await self.audio_queue.get()
        return frame

    async def get_video_frame(self) -> VideoFrame:
        frame = await self.video_queue.get()
        self.consumed_frame_count += 1
        return frame

    def _get_start_time(self):
        # unsafe
        if self.start_time == 0:
            self.start_time = time.time()
        return self.start_time

    async def next_audio_timestamp(self):
        self.audio_timestamp += int(self.sample_rate * self.audio_ptime)
        self.audio_frame_count += 1
        start_time = self._get_start_time()
        wait = start_time + self.audio_frame_count * self.audio_ptime - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self.audio_timestamp, self.audio_timebase

    async def next_video_timestamp(self):
        self.video_timestamp += int(self.clock_rate * self.video_ptime)
        self.video_frame_count += 1
        start_time = self._get_start_time()
        wait = start_time + self.video_frame_count * self.video_ptime - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self.video_timestamp, self.video_timebase


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, config: PlayerConfig, track_sync: StreamTrackSync):
        super().__init__()
        self.config = config
        self.track_sync = track_sync


    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.track_sync.get_audio_frame()
        frame: AudioFrame
        pts, timebase = await self.track_sync.next_audio_timestamp()
        frame.pts = pts
        frame.time_base = timebase
        return frame

    def stop(self):
        super().stop()


class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config: PlayerConfig, track_sync: StreamTrackSync):
        super().__init__()
        self.config = config
        self.track_sync = track_sync

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.track_sync.get_video_frame()
        frame: VideoFrame
        pts, timebase = await self.track_sync.next_video_timestamp()
        frame.pts = pts
        frame.time_base = timebase
        return frame

    def stop(self):
        super().stop()

