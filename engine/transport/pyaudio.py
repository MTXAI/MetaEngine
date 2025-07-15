import numpy as np
import pyaudio

from engine.config import PlayerConfig
from engine.transport import Transport


class TransportPyAudio(Transport):
    kind: str = "pyaudio"
    audio_only: bool = True
    wait: bool = True

    def __init__(self, config: PlayerConfig):
        self.rate = config.sample_rate
        self.format = format
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            output=True,
            frames_per_buffer=int(self.rate / config.fps),
        )

    async def put_audio_frame(self, frame: np.ndarray):
        if self.stream is not None and self.stream.is_active():
            audio_data = frame.tobytes()
            self.stream.write(audio_data)

    async def put_video_frame(self, frame: np.ndarray):
        pass

    def stop(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

