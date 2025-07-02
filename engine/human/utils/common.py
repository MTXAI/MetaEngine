import logging

import numpy as np
import resampy
import soundfile


def resample_sound(self, byte_stream, resample_rate=16000):
    stream, sample_rate = soundfile.read(byte_stream)  # [T*sample_rate,] float64
    stream = stream.astype(np.float32)

    if stream.ndim > 1:
        logging.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
        stream = stream[:, 0]

    if sample_rate != self.sample_rate and stream.shape[0] > 0:
        logging.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
        stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=resample_rate)

    return stream
