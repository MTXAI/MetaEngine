import hashlib
import json
import os
from ast import literal_eval
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import yaml
from multimethod import multimethod

from engine.utils.common import EasyDict


class EasyConfig(EasyDict):
    def load(self, fpath: str, *, recursive: bool = False) -> None:
        """load cfg from yaml

        Args:
            fpath (str): path to the yaml file
            recursive (bool, optional): recursily load its parent defaul yaml files. Defaults to False.
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        if recursive:
            extension = os.path.splitext(fpath)[1]
            while os.path.dirname(fpath) != fpath:
                fpath = os.path.dirname(fpath)
                fpaths.append(os.path.join(fpath, 'default' + extension))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath) as f:
                    self.update(yaml.safe_load(f))

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    # mutimethod makes python supports function overloading
    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], EasyConfig):
                    self[key] = EasyConfig()
                # recursively update
                self[key].update(value)
            else:
                self[key] = value

    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith('--'):
                opt = opt[2:]
            if '=' in opt:
                key, value = opt.split('=', 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split('.')
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, EasyConfig())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                value = value.dict()
            configs[key] = value
        return configs

    def hash(self) -> str:
        buffer = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                seperator = '\n'
            else:
                seperator = ' '
            text = key + ':' + seperator + str(value)
            lines = text.split('\n')
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (' ' * 2) + line
            texts.extend(lines)
        return '\n'.join(texts)

    def load_args(self, args):
        for arg_key, arg_value in args.__dict__.items():
            self.__setattr__(arg_key, arg_value)

    def save(self, path):
        cfg = deepcopy(self)
        for k, v in cfg.items():
            if not callable(v.__str__):
                cfg.dict()[k] = ''
        with open(path, 'w') as f:
            yaml.dump(cfg.dict(), f, sort_keys=True)
