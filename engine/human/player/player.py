import asyncio
import time
from typing import Tuple

from langchain_openai import ChatOpenAI

from engine.agent.agents.base_agent import BaseAgent
from engine.config import PlayerConfig
from engine.human.avatar.avatar import AvatarModelWrapper
from engine.human.player.container import HumanContainer
from engine.human.player.state import *
from engine.human.player.track import AudioStreamTrack, VideoStreamTrack, StreamTrackSync
from engine.human.voice.voice import TTSModelWrapper
from engine.runtime import thread_pool
from engine.utils.concurrent.pool import TaskInfo


class HumanPlayer:
    def __init__(
            self,
            config: PlayerConfig,
            agent: BaseAgent,
            tts_model: TTSModelWrapper,
            avatar_model: AvatarModelWrapper,
            avatar: Tuple,
            loop: asyncio.AbstractEventLoop,
    ):
        self.config = config
        self.track_sync = StreamTrackSync(
            config,
        )
        self.audio_track = AudioStreamTrack(
            config,
            self.track_sync,
        )
        self.video_track = VideoStreamTrack(
            config,
            self.track_sync,
        )
        shared_state = HumanState(state=StateReady)
        self.state = shared_state
        self.container = HumanContainer(
            config,
            agent,
            tts_model,
            avatar_model,
            avatar,
            self.track_sync,
            loop,
        )
        self.loop = loop
        self._start = False
        self._speaking = False

    def busy(self):
        return self.container.get_state() == StateBusy

    def flush(self):
        self.container.flush()

    def start(self):
        if self._start:
            return
        thread_pool.submit(
            self.container.process_text_data_worker,
            task_info=TaskInfo(
                name=f"container.process_text_data_worker"
            )
        )
        thread_pool.submit(
            self.container.process_audio_data_worker,
            task_info=TaskInfo(
                name=f"container.process_audio_data_worker"
            )
        )
        self._start = True

    def shutdown(self):
        self.container.shutdown()


if __name__ == '__main__':

    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.utils.data import Data
    from engine.config import ONE_API_LLM_MODEL
    from engine.human.voice.tts_ali import AliTTSWrapper
    from engine.human.voice.tts_edge import EdgeTTSWrapper
    from engine.agent.agents.custom import SimpleAgent

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

    # llm_model = ChatOpenAI(
    #     model=QWEN_LLM_MODEL.model_id,
    #     api_key=QWEN_LLM_MODEL.api_key,
    #     base_url=QWEN_LLM_MODEL.api_base_url,
    # )
    llm_model = ChatOpenAI(
        model=ONE_API_LLM_MODEL.model_id,
        api_key=ONE_API_LLM_MODEL.api_key,
        base_url=ONE_API_LLM_MODEL.api_base_url,
    )
    agent = SimpleAgent(llm_model)

    # vector_store = try_load_db(DEFAULT_PROJECT_CONFIG.vecdb_path, DEFAULT_PROJECT_CONFIG.docs_path)
    # agent = KnowledgeAgent(llm_model, vector_store)

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
        for i in range(1):
            player.flush()
            res_data = player.container.put_text_data(Data(
                data="介绍故宫",
                is_chat=True,
                stream=True,
            ))
            print(res_data)
            time.sleep(5)

    asyncio.run_coroutine_threadsafe(listen_audio(), loop=loop)
    asyncio.run_coroutine_threadsafe(listen_video(), loop=loop)
    asyncio.run_coroutine_threadsafe(put_text_data(), loop=loop)
    asyncio.set_event_loop(loop)
    loop.run_forever()
