import asyncio
import queue
import time
from typing import Tuple, Callable

from openai import AsyncOpenAI

from engine.config import PlayerConfig
from engine.human.avatar.avatar import AvatarModelWrapper
from engine.human.player.container import AudioContainer, VideoContainer, TextContainer
from engine.human.player.track import AudioStreamTrack, VideoStreamTrack, StreamTrackSync
from engine.human.voice.voice import TTSModelWrapper
from engine.utils.pipeline import Pipeline, TODOPipelineCallback
from engine.utils.pool import TaskInfo
from engine.runtime import thread_pool


class HumanPlayer:
    def __init__(
            self,
            config: PlayerConfig,
            agent: AsyncOpenAI,
            tts_model: TTSModelWrapper,
            avatar_model: AvatarModelWrapper,
            avatar: Tuple,
            loop: asyncio.AbstractEventLoop,
    ):
        self.config = config
        self.track_sync = StreamTrackSync(config)
        self.audio_track = AudioStreamTrack(
            config,
            self.track_sync,
        )
        self.video_track = VideoStreamTrack(
            config,
            self.track_sync,
        )
        self.text_container = TextContainer(
            config,
            agent,
            tts_model
        )
        self.audio_container = AudioContainer(
            config,
            avatar_model,
            self.track_sync,
            loop
        )
        self.video_container = VideoContainer(
            config,
            avatar_model,
            avatar,
            self.track_sync,
            loop
        )
        self.containers = [
            self.text_container,
            self.audio_container,
            self.video_container,
        ]
        self.loop = loop
        pipeline_callback = TODOPipelineCallback()
        self.pipelines = [
            Pipeline(
                name="TextPipeline",
                producer=self.text_container.text_producer,
                consumer=self.text_container.text_consumer,
                callback=pipeline_callback,
            ),
            Pipeline(
                name="AudioPipeline",
                producer=self.text_container.audio_producer,
                consumer=self.audio_container.audio_consumer,
                callback=pipeline_callback,
            ),
            Pipeline(
                name="VideoPipeline",
                producer=self.audio_container.audio_feature_producer,
                consumer=self.video_container.audio_feature_consumer,
                callback=pipeline_callback,
            )
        ]
        self._start = False

    def flush(self):
        for container in self.containers:
            container.flush()

    def start(self):
        if self._start:
            return
        for i, pipe in enumerate(self.pipelines):
            thread_pool.submit(
                pipe.produce_worker,
                task_info=TaskInfo(
                    name=f"{i}_produce_worker"
                )
            )
            thread_pool.submit(
                pipe.consume_worker,
                task_info=TaskInfo(
                    name=f"{i}_consume_worker"
                )
            )
        self._start = True

    def shutdown(self):
        for container in self.containers:
            container.shutdown()
        for pipe in self.pipelines:
            pipe.shutdown()


if __name__ == '__main__':

    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.utils.data import Data
    from engine.human.voice.tts_edge import EdgeTTSWrapper
    from engine.human.voice.tts_ali import AliTTSWrapper

    a_f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr.wav'
    c_f = '../../../checkpoints/wav2lip.pth'

    # 创建Player实例并启动
    loop = asyncio.new_event_loop()

    tts_model = AliTTSWrapper(
        model_str="cosyvoice-v1",
        api_key="sk-361f246a74c9421085d1d137038d5064",
        voice_type="longxiaochun",
        sample_rate=WAV2LIP_PLAYER_CONFIG.sample_rate,
    )

    # tts_model = EdgeTTSWrapper(
    #     voice_type="zh-CN-YunxiaNeural",
    #     sample_rate=WAV2LIP_PLAYER_CONFIG.sample_rate,
    # )
    avatar_model = Wav2LipWrapper(c_f)
    agent = AsyncOpenAI(
        # one api 生成的令牌
        api_key="sk-A3DJFMPvXa7Ot9faF4882708Aa2b419c87A50fFe8223B297",
        base_url="http://localhost:3000/v1"
    )
    player = HumanPlayer(
        config=WAV2LIP_PLAYER_CONFIG,
        agent=agent,
        tts_model=tts_model,
        avatar_model=avatar_model,
        avatar=load_avatar(a_f),
        loop=loop,
    )

    player.start()

    async def listen_audio():
        i = 0
        counttime = 0
        t = time.perf_counter()
        while True:
            await asyncio.sleep(0.01)
            frame = await player.audio_track.recv()
            counttime += (time.perf_counter() - t)
            i += 1
            if i >= 100:
                print(f"{i}, {i / counttime}: {frame}, {player.track_sync.audio_queue.qsize()}")
                i = 0
                counttime = 0

    async def listen_video():
        i = 0
        counttime = 0
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            frame = await player.video_track.recv()
            counttime += (time.perf_counter() - t)
            i += 1
            if i >= 100:
                print(f"{i}, {i / counttime}: {frame}, {player.track_sync.video_queue.qsize()}")
                i = 0
                counttime = 0

    async def put_text_data():
        await asyncio.sleep(1)
        res_data = player.text_container.put_text_data(Data(
            data="你好, 我是墨菲",
            is_chat=False,
        ))
        print(res_data)

    # asyncio.run_coroutine_threadsafe(listen_audio(), loop=loop)
    # asyncio.run_coroutine_threadsafe(listen_video(), loop=loop)
    asyncio.run_coroutine_threadsafe(put_text_data(), loop=loop)
    asyncio.set_event_loop(loop)
    loop.run_forever()
