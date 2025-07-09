import io
import logging
from typing import Union

import numpy as np
import resampy
import soundfile


def read_sound(file_like: Union[io.BytesIO, bytes, str]):
    if isinstance(file_like, io.BytesIO):
        file_like.seek(0)
    if isinstance(file_like, bytes):
        file_like = io.BytesIO(file_like)
    stream, sample_rate = soundfile.read(file_like)
    stream = stream.astype(np.float32)
    return stream, sample_rate


def resample_sound(file_like: Union[io.BytesIO, bytes, str], resample_rate: int=16000):
    stream, sample_rate = read_sound(file_like)
    if stream.ndim > 1:
        stream = stream[:, 0]
    if sample_rate != resample_rate and stream.shape[0] > 0:
        stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=resample_rate)
    return stream


def resample_sound_raw(file_like: Union[io.BytesIO, bytes, str], resample_rate: int=16000, subtype: str="PCM_16"):
    if isinstance(file_like, io.BytesIO):
        file_like.seek(0)
    if isinstance(file_like, bytes):
        file_like = io.BytesIO(file_like)
    stream, sample_rate = soundfile.read(
        file=file_like,
        dtype='float32',
        format='RAW',
        subtype=subtype,
        samplerate=resample_rate,
        channels=1,
        endian='FILE',
    )
    stream = stream.astype(np.float32)
    return stream
