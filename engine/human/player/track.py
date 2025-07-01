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


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, config: PlayerConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.ptime = config.audio_ptime
        self.timebase = fractions.Fraction(1, self.sample_rate)
        self.timestamp = 0
        self.queue = asyncio.Queue()

    async def put_frame(self, frame: AudioFrame):
        await self.queue.put(frame)

    def next_timestamp(self):
        self.timestamp += int(self.ptime * self.sample_rate)
        return self.timestamp, self.timebase

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.queue.get()
        frame: AudioFrame
        pts, timebase = self.next_timestamp()
        frame.pts = pts
        frame.time_base = timebase
        return frame

    def stop(self):
        super().stop()


class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config: PlayerConfig):
        super().__init__()
        self.config = config
        self.clock_rate = config.video_clock_rate
        self.ptime = config.video_ptime
        self.timebase = fractions.Fraction(1, self.clock_rate)
        self.timestamp = 0
        self.queue = asyncio.Queue()

    async def put_frame(self, frame: VideoFrame):
        await self.queue.put(frame)

    def next_timestamp(self):
        self.timestamp += int(self.ptime * self.clock_rate)
        return self.timestamp, self.timebase

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.queue.get()
        frame: VideoFrame
        pts, timebase = self.next_timestamp()
        frame.pts = pts
        frame.time_base = timebase
        return frame

    def stop(self):
        super().stop()

