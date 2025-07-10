import copy
import pickle
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.multiprocessing as mp


def get_file_path(__file):
    return Path(__file)


class EasyDict(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def __deepcopy__(self, name):
        copy_dict = dict()
        for key, value in self.items():
            if hasattr(value, '__deepcopy__'):
                copy_dict[key] = copy.deepcopy(value)
            else:
                copy_dict[key] = value
        return self.__class__(copy_dict)

    def __getstate__(self):
        return pickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = pickle.loads(state)

    def __exists__(self, name):
        return name in self.__dict__

    def __str__(self) -> str:
        text = ''
        for key, value in self.items():
            if len(text) > 0:
                text += ', '
            if hasattr(value, '__str__'):
                value_str = str(value)
            else:
                value_str = f"{type(value).__name__}_{id(value)}"
            text += key + ': ' + value_str
        return '{%s}' % text


def get_device_and_start_method() -> Tuple[str, str]:
    device = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu")
    if device == 'cuda':
        start_method = 'spawn'
    elif device == 'mps':
        start_method = 'spawn'
    else:
        start_method = 'spawn'

    if mp.get_start_method(allow_none=True) != start_method:
        mp.set_start_method(start_method, force=True)

    return device, start_method


def check_fp16_support():
    if not torch.cuda.is_available():
        return False

    device = torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 6:
        try:
            _ = torch.tensor([1.0], device="cuda", dtype=torch.float16)
            return True
        except:
            return False
    else:
        return False

