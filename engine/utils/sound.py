import io
import logging
from typing import Union

import numpy as np
import resampy
import soundfile


def resample_sound(file_like: Union[io.BytesIO, str], resample_rate: int=16000):
    if isinstance(file_like, io.BytesIO):
        file_like.seek(0)
    stream, sample_rate = soundfile.read(file_like)  # [T*sample_rate,] float64
    stream = stream.astype(np.float32)

    if stream.ndim > 1:
        logging.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
        stream = stream[:, 0]

    if sample_rate != resample_rate and stream.shape[0] > 0:
        # logging.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {resample_rate}.')
        stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=resample_rate)

    return stream
