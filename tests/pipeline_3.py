## pipeline 语音 - 文字 - 文字

import asyncio
from pathlib import Path

from funasr import AutoModel

from engine.utils.pipeline import AsyncPipeline, AsyncConsumerFactory

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
    print(total_chunk_num)
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        yield {
            "speech": speech_chunk,
            "is_final": is_final,
        }

cache = {}
def consume_fn_1(data, processed_data=None):
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
    print(res[0]['text'])
    return {
        "content": res[0]['text'],
        "is_final": is_final,
    }


from openai import AsyncOpenAI
client = AsyncOpenAI(
            # one api 生成的令牌
            api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
            base_url="http://localhost:3000/v1"
        )


async def consume_fn_2(data, processed_data=None):
    is_final = data["is_final"]
    if not is_final:
        return
    content = data["content"]
    chat_completion = await client.chat.completions.create(
        # model="doubao-1.5-lite-32k",
        model="qwen-turbo",
        messages=[
            {
                "role": "user",
                "content": f"请简要总结: {content}",
            }
        ],
        stream=True,
    )
    async for chunk in chat_completion:
        c = chunk.choices[0].delta.content
        # yield c
        print(c)


all_content = ""
def handler_2(content):
    global all_content
    ## todo 判断 content 是否需要转文字
    all_content += content['content'].strip()
    return {
        "content": all_content,
        "is_final": content["is_final"],
    }

async def run_pipeline():
    pipeline = AsyncPipeline(
        producer=produce,
        consumers=[
            AsyncConsumerFactory.with_consume_fn(consume_fn_1),
            AsyncConsumerFactory.with_consume_fn(consume_fn_2),
        ],
    )

    tasks = [
        pipeline.start(),
        pipeline.stop(),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(run_pipeline())

