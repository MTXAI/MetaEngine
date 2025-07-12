import asyncio
import fractions
import time
from typing import Union

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame
from av.frame import Frame
from av.packet import Packet

from engine.config import PlayerConfig
from engine.transport import Transport


class AudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, config: PlayerConfig):
        super().__init__()
        self.config = config

        self.queue = asyncio.Queue(config.fps)

        self.rate = config.sample_rate
        self.ptime = config.audio_ptime
        self.timebase = fractions.Fraction(1, self.rate)

        self.frame_count = 0
        self._timestamp = 0
        self._start_time = 0.0

    async def put_frame(self, frame: AudioFrame):
        await self.queue.put(frame)

    async def get_frame(self) -> AudioFrame:
        frame = await self.queue.get()
        return frame

    async def next_timestamp(self):
        if self._start_time == 0:
            self._start_time = time.time()
            self._timestamp = 0
            return self._timestamp
        self._timestamp += int(self.rate * self.ptime)
        self.frame_count += 1
        wait = self._start_time + self.frame_count * self.ptime - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self._timestamp

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.get_frame()
        frame: AudioFrame
        pts = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = self.timebase
        return frame

    def stop(self):
        super().stop()


class VideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, config: PlayerConfig):
        super().__init__()
        self.config = config

        self.queue = asyncio.Queue(config.fps // config.frame_multiple)

        self.rate = config.clock_rate
        self.ptime = config.video_ptime
        self.timebase = fractions.Fraction(1, self.rate)

        self.frame_count = 0
        self._timestamp = 0
        self._start_time = 0.0

    async def put_frame(self, frame: VideoFrame):
        await self.queue.put(frame)

    async def get_frame(self) -> VideoFrame:
        frame = await self.queue.get()
        return frame

    async def next_timestamp(self):
        if self._start_time == 0:
            self._start_time = time.time()
            self._timestamp = 0
            return self._timestamp
        self._timestamp += int(self.rate * self.ptime)
        self.frame_count += 1
        wait = self._start_time + self.frame_count * self.ptime - time.time()
        if wait > 0:
            await asyncio.sleep(wait)
        return self._timestamp

    async def recv(self) -> Union[Frame, Packet]:
        frame = await self.get_frame()
        frame: VideoFrame
        pts = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = self.timebase
        return frame

    def stop(self):
        super().stop()

class TransportWebRTC(Transport):
    kind: str = "webrtc"
    def __init__(self, config: PlayerConfig):
        super().__init__()
        self.audio_track = AudioStreamTrack(config)
        self.video_track = VideoStreamTrack(config)
        self.sample_rate = config.sample_rate
        self.clock_rate = config.clock_rate

    async def put_audio_frame(self, frame: np.ndarray):
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = self.sample_rate
        await self.audio_track.put_frame(new_frame)

    async def put_video_frame(self, frame: np.ndarray):
        frame[0, :] &= 0xFE  # 确保第一行是偶数，避免某些视频问题
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        await self.video_track.put_frame(new_frame)

    async def start(self):
        pass

    def stop(self):
        self.audio_track.stop()
        self.video_track.stop()
