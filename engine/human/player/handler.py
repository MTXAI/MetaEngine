import logging
from typing import Any

from engine.human.player.data import TextData


def log_handler(data: Any):
    logging.info(f"got data {data}")


def text_fileter_handler(data: TextData):
    pass

