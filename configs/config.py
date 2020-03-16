from pathlib import Path
import json
from typing import Union

from _jsonnet import evaluate_file


class Config:
    def __init__(self):
        raise NotImplementedError

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == ".json":
            with open(path, "r") as fp:
                data = json.load(fp)
        elif path.suffix == ".jsonnet":
            data = json.loads(evaluate_file(str(path)))
        else:
            raise ValueError(f"Invalid config file: {path.suffix}")

        assert isinstance(data, dict), f"Config file must be in dictionary-like format!"
        config = cls(**data)
        return config

    def __repr__(self):
        return str(self.__dict__)
