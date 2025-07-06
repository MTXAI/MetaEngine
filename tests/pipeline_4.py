## pipeline 语音 - 文字 - 文字 - 语音

import asyncio
from pathlib import Path

import dashscope
import pyaudio
from dashscope.audio.tts_v2 import *
from engine.utils.pipeline import AsyncPipeline, AsyncBridgeConsumer, PipelineCallback, \
    AsyncConsumerFactory
from funasr import AutoModel

# 若没有将API Key配置到环境变量中，需将下面这行代码注释放开，并将apiKey替换为自己的API Key
dashscope.api_key = "sk-361f246a74c9421085d1d137038d5064"
model = "cosyvoice-v1"
voice = "longxiaochun"
class Callback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        # print("websocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050, output=True
        )

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        # stop player
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        # print(f"recv speech synthsis message {message}")
        pass

    def on_data(self, data: bytes) -> None:
        # print("audio result length:", len(data))
        self._stream.write(data)
callback = Callback()
synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=callback,
    )


# chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
chunk_size = [0, 100, 50]
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming", hub='ms', disable_update=True)

import soundfile


def get_file_path(__file):
    return Path(__file)

async def producer():
    wav_path = get_file_path(__file__).parent / "test_datas/asr_example.wav"
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
def consume_fn_1(data):
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


all_content = ""
def handler_2(content):
    global all_content
    ## todo 判断 content 是否需要转文字
    all_content += content['content'].strip()
    return {
        "content": all_content,
        "is_final": content["is_final"],
    }

async def bridge_producer(data):
    is_final = data["is_final"]
    if not is_final:
        return
    chat_completion = await client.chat.completions.create(
        # model="doubao-1.5-lite-32k",
        model="qwen-turbo",
        messages=[
            {
                "role": "user",
                "content": f"请简要总结: {data['content']}",
            }
        ],
        stream=True,
    )
    async for chunk in chat_completion:
        content = chunk.choices[0].delta.content
        print('prod', content)
        yield content

stop_event = asyncio.Event()
callback = PipelineCallback.with_events(stop_event=stop_event)
pipeline_bridge = AsyncBridgeConsumer(
    stop_event=stop_event,
    generator=bridge_producer,
)

def consume_fn_3(content):
    print('cons', content)
    synthesizer.streaming_call(content)
    return content


class TTSCallback(PipelineCallback):
    def on_stop(self):
        synthesizer.streaming_complete()

async def run_pipeline():
    pipeline = AsyncPipeline(
        producer=producer,
        consumers=[
            AsyncConsumerFactory.with_consume_fn(consume_fn_1),
            pipeline_bridge,
        ],
        callback=callback,
    )

    pipeline2 = AsyncPipeline(
            producer=pipeline_bridge.to_producer(),
            consumers=[
                AsyncConsumerFactory.with_consume_fn(consume_fn_3),
            ],
            callback=TTSCallback(),
        )

    tasks = [
        pipeline.start(),
        pipeline2.start(),
        pipeline.stop(),
        pipeline2.stop(),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(run_pipeline())
