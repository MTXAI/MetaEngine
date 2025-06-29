import copy
import pickle
from typing import Any


class Data(dict):
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
        return Data(copy_dict)

    def __getstate__(self):
        return pickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = pickle.loads(state)

    def __exists__(self, name):
        return name in self.__dict__

    def __str__(self):
        return str(self.data)

