import asyncio
import logging
import time
from typing import Union, List, Tuple

from langchain_openai import ChatOpenAI

from engine.agent.agents.base_agent import BaseAgent
from engine.config import PlayerConfig
from engine.human.avatar import AvatarModelWrapper
from engine.human.avatar.avatar import AvatarProcessor, Avatar
from engine.human.player.container import HumanContainer
from engine.human.player.state import *
from engine.human.transport import Transport
from engine.human.voice import TTSModelWrapper
from engine.human.voice.voice import VoiceProcessor
from engine.runtime import thread_pool
from engine.utils.concurrent.pool import TaskInfo
from engine.utils import Data


class HumanPlayer:
    def __init__(
            self,
            config: PlayerConfig,
            agent: BaseAgent,
            tts_model: TTSModelWrapper,
            avatar: Avatar,
            avatar_model: AvatarModelWrapper,
            voice_processor: VoiceProcessor,
            avatar_processor: AvatarProcessor,
            loop: asyncio.AbstractEventLoop,
            main_transport: Transport,
            other_transports: Union[Transport, List[Transport], Tuple[Transport]]=None,
    ):
        self.config = config
        self.container = HumanContainer(
            self.config,
            agent,
            tts_model,
            avatar,
            avatar_model,
            voice_processor,
            avatar_processor,
            loop,
            main_transport,
            other_transports,
        )
        self._start = False

    def is_ready(self):
        return self.container.get_state() == StateReady

    def is_busy(self):
        return self.container.get_state() == StateBusy or self.container.get_state() == StatePause

    def set_agent(self, agent: BaseAgent) -> bool:
        # agent 正在使用中
        if self.container.get_state() == StateBusy:
            return False
        self.container.agent = agent
        return True

    def set_tts_model(self, tts_model: TTSModelWrapper) -> bool:
        # tts model 正在使用中
        if self.container.get_state() == StateBusy:
            return False
        self.container.tts_model = tts_model
        return True

    def pause(self):
        self.container.pause()

    def put_text_data(self, data: Data):
        return self.container.put_text_data(data)

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
        thread_pool.submit(
            self.container.process_frames_worker,
            task_info=TaskInfo(
                name=f"container.process_frames_worker"
            )
        )
        self._start = True

    def shutdown(self):
        self.container.shutdown()
        self._start = False


if __name__ == '__main__':

    from engine.utils.data import Data
    from engine.config import ONE_API_LLM_MODEL
    from engine.human.voice.tts_ali import AliTTSWrapper
    from engine.human.voice.tts_edge import EdgeTTSWrapper
    from engine.agent.agents.custom import SimpleAgent
    from engine.utils import get_file_path
    from engine.human.avatar import wav2lip
    from engine.config import DEFAULT_VOICE_PROCESSOR_CONFIG, DEFAULT_AVATAR_PROCESSOR_CONFIG, WAV2LIP_PLAYER_CONFIG
    from engine.human.transport import Transport, TransportWebRTC

    a_f = '../../../avatars/wav2lip256_avatar1'
    a_p = get_file_path(a_f)
    c_f = '../../../checkpoints/wav2lip/wav2lip.pth'
    c_p = get_file_path(c_f)

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

    avatar = wav2lip.load_avatar(a_p.absolute().as_posix())
    avatar_model = wav2lip.Wav2LipWrapper(c_p.absolute().as_posix(), avatar)

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

    voice_processor = VoiceProcessor(DEFAULT_VOICE_PROCESSOR_CONFIG)
    avatar_processor = AvatarProcessor(DEFAULT_AVATAR_PROCESSOR_CONFIG)

    webrtc_transport = TransportWebRTC(WAV2LIP_PLAYER_CONFIG)

    player = HumanPlayer(
        config=WAV2LIP_PLAYER_CONFIG,
        agent=agent,
        tts_model=tts_model,
        avatar=avatar,
        avatar_model=avatar_model,
        voice_processor=voice_processor,
        avatar_processor=avatar_processor,
        loop=loop,
        main_transport=webrtc_transport,
        other_transports=None,
    )

    player.start()

    async def listen_audio():
        i = 0
        counttime = 0
        t = time.perf_counter()
        while True:
            await asyncio.sleep(0.01)
            frame = await webrtc_transport.audio_track.recv()
            counttime += (time.perf_counter() - t)
            i += 1
            if i >= 100:
                logging.info(f"{i}, {i / counttime}: {frame}, {webrtc_transport.audio_track.queue.qsize()}")
                i = 0
                counttime = 0

    async def listen_video():
        i = 0
        counttime = 0
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            frame = await webrtc_transport.video_track.recv()
            counttime += (time.perf_counter() - t)
            i += 1
            if i >= 100:
                logging.info(f"{i}, {i / counttime}: {frame}, {webrtc_transport.video_track.queue.qsize()}")
                i = 0
                counttime = 0

    async def put_text_data():
        for i in range(1):
            res_data = player.put_text_data(Data(
                data="介绍故宫",
                is_chat=True,
                stream=True,
            ))
            logging.info(res_data)
            time.sleep(5)

    asyncio.run_coroutine_threadsafe(listen_audio(), loop=loop)
    asyncio.run_coroutine_threadsafe(listen_video(), loop=loop)
    asyncio.run_coroutine_threadsafe(put_text_data(), loop=loop)
    asyncio.set_event_loop(loop)
    loop.run_forever()
