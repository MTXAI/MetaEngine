from typing import Any, Dict

from numpy import ndarray


class Data:
    data = {}

    def is_stream(self) -> bool:
        return self.data.get("is_stream", False)

    def is_final(self) -> bool:
        return self.data.get("is_final", True)

    def get(self, k) -> Any:
        return self.data.get(k, None)

    def set(self, k, v):
        self.data[k] = v

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
    def __init__(self, sound: ndarray, stream: bool=False, final: bool=True):
        self.data = {
            "sound": sound,
            "is_stream": stream,
            "is_final": final,
        }


class AvatarData(Data):
    def __init__(self, avatar: ndarray, stream: bool=False, final: bool=True, **kwargs):
        self.data = {
            "avatar": avatar,
            "is_stream": stream,
            "is_final": final,
        }
        for k, v in kwargs.items():
            self.data[k] = v


class VideoData(Data):
    pass

