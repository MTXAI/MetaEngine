from typing import Any, Dict

from numpy import ndarray


class Data:
    data: Dict[str: Any] = {}

    def is_stream(self) -> bool:
        return self.data.get("is_stream", False)

    def is_final(self) -> bool:
        return self.data.get("is_final", True)

    def __str__(self):
        return str(self.data)


class TextData(Data):
    def __init__(self, text: str, stream: bool=False, final: bool=True):
        self.data = {
            "text": text,
            "is_stream": stream,
            "is_final": final,
        }


class SoundData(Data):
    def __init__(self, speech: ndarray, stream: bool=False, final: bool=True):
        self.data = {
            "speech": speech,
            "is_stream": stream,
            "is_final": final,
        }


class VideoData(Data):
    pass

