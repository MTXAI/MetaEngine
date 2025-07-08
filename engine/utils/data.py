from typing import Any

from engine.utils.common import EasyDict


class Data(EasyDict):
    def set(self, key: str, value: Any):
        self[key] = value

