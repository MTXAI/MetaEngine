import numpy as np


class Transport:
    kind: str = ""
    audio_only: bool = False
    wait: bool = False

    async def put_audio_frame(self, frame: np.ndarray):
        pass

    async def put_video_frame(self, frame: np.ndarray):
        pass

    def shutdown(self):
        pass
