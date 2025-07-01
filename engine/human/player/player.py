import asyncio
import time
from typing import Tuple, Callable

from engine.config import PlayerConfig
from engine.human.avatar.avatar import ModelWrapper
from engine.human.player.container import AudioContainer, VideoContainer
from engine.human.player.track import AudioStreamTrack, VideoStreamTrack, StreamTrackSync
from engine.utils.pipeline import Pipeline


class HumanPlayer:
    def __init__(
            self,
            config: PlayerConfig,
            model: ModelWrapper,
            avatar: Tuple,
            loop: asyncio.AbstractEventLoop,
            audio_producer: Callable,
    ):
        self.config = config
        self.model = model
        self.track_sync = StreamTrackSync(config)
        self.audio_track = AudioStreamTrack(
            config,
            self.track_sync,
        )
        self.video_track = VideoStreamTrack(
            config,
            self.track_sync,
        )
        self.audio_container = AudioContainer(
            config,
            model,
            self.track_sync,
            loop
        )
        self.video_container = VideoContainer(
            config,
            model,
            avatar,
            self.track_sync,
            loop
        )
        self.loop = loop
        self.pipelines = [
            Pipeline(
                producer=audio_producer,
                consumer=self.audio_container.consumer
            ),
            Pipeline(
                producer=self.audio_container.producer,
                consumer=self.video_container.consumer
            )
        ]
        self._start = False


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
        self.audio_container.shutdown()
        for pipe in self.pipelines:
            pipe.shutdown()


if __name__ == '__main__':

    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.runtime import thread_pool
    from engine.utils.pool import TaskInfo
    from engine.human.voice.asr import soundfile_producer

    f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr.wav'
    c_f = '../../../checkpoints/wav2lip.pth'
    model = Wav2LipWrapper(c_f)

    # 创建Player实例并启动
    loop = asyncio.new_event_loop()

    player = HumanPlayer(
        config=WAV2LIP_PLAYER_CONFIG,
        model=model,
        avatar=load_avatar(f),
        loop=loop,
        audio_producer=soundfile_producer(s_f, fps=10)
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

    asyncio.run_coroutine_threadsafe(listen_audio(), loop=loop)
    asyncio.run_coroutine_threadsafe(listen_video(), loop=loop)
    asyncio.set_event_loop(loop)
    loop.run_forever()
