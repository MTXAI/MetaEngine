import numpy as np


class Transport:
    kind: str = ""

    async def put_audio_frame(self, frame: np.ndarray):
        pass

    async def put_video_frame(self, frame: np.ndarray):
        pass

    def is_ready(self, frame_index) -> bool:
        pass

    def start(self):
        pass

    def stop(self):
        pass
