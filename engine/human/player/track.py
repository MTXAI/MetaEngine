import asyncio
import fractions
import logging
import sys
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

        self.audio_queue = asyncio.Queue(self.fps * 10)
        self.video_queue = asyncio.Queue(self.fps * 10)
        self.audio_frame: AudioFrame = None
        self.video_frame: VideoFrame = None

    async def put_audio_frame(self, frame: AudioFrame):
        await self.audio_queue.put(frame)

    async def put_video_frame(self, frame: VideoFrame):
        await self.video_queue.put(frame)

    async def _sync_frame(self):
        if self.audio_frame is None:
            self.audio_frame = await self.audio_queue.get()
        if self.video_frame is None:
            self.video_frame = await self.video_queue.get()

    async def get_audio_frame(self) -> AudioFrame:
        await self._sync_frame()
        frame = self.audio_frame
        self.audio_frame = None
        return frame

    async def get_video_frame(self) -> VideoFrame:
        await self._sync_frame()
        frame = self.video_frame
        self.video_frame = None
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
        self.timestamp = 0
        self.track_sync = track_sync

        self.start_time = 0.0
        self.frame_count = 0

    async def next_timestamp(self):
        self.timestamp += int(self.ptime * self.sample_rate)
        self.frame_count += 1
        if self.start_time == 0:
            self.start_time = time.time()
        else:
            wait_time = self.start_time + self.frame_count * self.ptime - time.time()
            await asyncio.sleep(wait_time)
            if self.frame_count >= self.fps:
                self.start_time = time.time()
                self.frame_count = 0
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
        self.ptime = config.video_ptime
        self.timebase = fractions.Fraction(1, self.clock_rate)
        self.timestamp = 0
        self.track_sync = track_sync

        self.start_time = 0.0
        self.frame_count = 0

    async def next_timestamp(self):
        self.timestamp += int(self.ptime * self.clock_rate)
        self.frame_count += 1
        if self.start_time == 0:
            self.start_time = time.time()
        else:
            wait_time = self.start_time + self.frame_count * self.ptime - time.time()
            await asyncio.sleep(wait_time)
            if self.frame_count >= self.fps:
                self.start_time = time.time()
                self.frame_count = 0
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

