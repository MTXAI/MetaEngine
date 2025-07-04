import asyncio
import fractions
import time
from typing import Union

from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame
from av.frame import Frame
from av.packet import Packet

from engine.config import PlayerConfig
from engine.utils.concurrent.queue import ObservableQueue


class StreamTrackSync:
    def __init__(self, config: PlayerConfig):
        self.fps = config.fps
        self.prefer = config.track_sync_prefer
        self.lock = asyncio.Lock()
        self.audio_queue = ObservableQueue(self.fps)
        self.video_queue = ObservableQueue(self.fps)

    def flush(self):
        self.audio_queue.clear()
        self.video_queue.clear()

    async def put_audio_frame(self, frame: AudioFrame):
        # if self.prefer == 'video':
        #     await self.video_queue.wait_for_data()
        await self.audio_queue.put(frame)

    async def put_video_frame(self, frame: VideoFrame):
        # if self.prefer == 'audio':
        #     await self.audio_queue.wait_for_data()
        await self.video_queue.put(frame)

    async def get_audio_frame(self) -> AudioFrame:
        frame = await self.audio_queue.get()
        return frame

    async def get_video_frame(self) -> VideoFrame:
        frame = await self.video_queue.get()
        return frame


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, config: PlayerConfig, track_sync: StreamTrackSync):
        super().__init__()
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.ptime = config.audio_ptime
        self.timebase = fractions.Fraction(1, self.sample_rate)
        self.track_sync = track_sync

        self.timestamp = 0
        self.start_time = 0.0
        self.frame_count = 0

    async def next_timestamp(self):
        self.timestamp += int(self.sample_rate * self.ptime)
        self.frame_count += 1
        if self.start_time == 0:
            self.start_time = time.time()
        else:
            wait = self.start_time + self.frame_count * self.ptime - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        if self.frame_count > self.fps:
            self.frame_count = 0
            self.start_time = time.time()
        return self.timestamp, self.timebase

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.track_sync.get_audio_frame()
        frame: AudioFrame
        pts, timebase = await self.next_timestamp()
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
        self.fps = config.fps
        self.clock_rate = config.video_clock_rate
        self.ptime = config.audio_ptime
        self.timebase = fractions.Fraction(1, self.clock_rate)
        self.track_sync = track_sync

        self.timestamp = 0
        self.start_time = 0.0
        self.frame_count = 0

    async def next_timestamp(self):
        self.timestamp += int(self.clock_rate * self.ptime)
        self.frame_count += 1
        if self.start_time == 0:
            self.start_time = time.time()
        else:
            wait = self.start_time + self.frame_count * self.ptime - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        if self.frame_count > self.fps:
            self.frame_count = 0
            self.start_time = time.time()
        return self.timestamp, self.timebase

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.track_sync.get_video_frame()
        frame: VideoFrame
        pts, timebase = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = timebase
        return frame

    def stop(self):
        super().stop()

