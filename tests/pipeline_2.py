## pipeline 语音 - 文字

import asyncio
from pathlib import Path

from engine.utils.pipeline import AsyncPipeline, AsyncConsumerFactory
from funasr import AutoModel

# chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
chunk_size = [0, 100, 50]
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", hub='ms', disable_update=True)

import soundfile


def get_file_path(__file):
    return Path(__file)

async def produce():
    wav_path = get_file_path(__file__).parent / "test_datas/asr.wav"
    wav_file = wav_path.absolute().as_posix()
    speech, sample_rate = soundfile.read(wav_file)
    chunk_stride = chunk_size[1] * 960  # 600ms

    total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        yield {
            "speech": speech_chunk,
            "is_final": is_final,
        }

cache = {}
def consume_fn(data, processed_data=None):
    speech_chunk = data["speech"]
    is_final = data["is_final"]
    res = model.generate(
        input=speech_chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
    )
    logging.info(res[0]['text'])
    return res[0]['text']


async def run_pipeline():
    pipeline = AsyncPipeline(
        producer=produce,
        consumers=[
            AsyncConsumerFactory.with_consume_fn(consume_fn)
        ],
    )

    tasks = [
        pipeline.start(),
        pipeline.stop(),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(run_pipeline())

