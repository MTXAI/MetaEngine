import asyncio
import logging
import time
from typing import Union, List, Tuple

from engine import runtime
from engine.config import PlayerConfig
from engine.human.character import Character
from engine.human.avatar import Avatar
from engine.human.player.container import HumanContainer
from engine.human.player.state import *
from engine.transport import Transport
from engine.human.voice import Voice
from engine.utils.concurrent import TaskInfo
from engine.utils import Data


class HumanPlayer:
    def __init__(
            self,
            config: PlayerConfig,
            character: Character,  # todo 直接由 config 构建,
            voice: Voice,
            avatar: Avatar,
            loop: asyncio.AbstractEventLoop,
            transports: List[Transport]=None,
    ):
        self.config = config
        self._state = HumanState(StateReady)
        self.container = HumanContainer(
            self.config,
            character,
            voice,
            avatar,
            self._state,
            loop,
            transports,
        )
        self._start = False

    def is_ready(self):
        return self._state.get_state() == StateReady

    def is_speaking(self):
        return self._state.get_state() == StateSpeaking

    def is_pause(self):
        return self._state.get_state() == StatePause

    def is_busy(self):
        return self._state.get_state() == StateBusy

    def state(self):
        return self._state.get_state()

    def add_transport(self, transport: Transport):
        if transport.kind in self.container.transports:
            logging.warning(f"Transport {transport.kind} already exists")
            return
        self.container.transports[transport.kind] = transport

    def remove_transport(self, kind: str):
        if kind not in self.container.transports:
            logging.warning(f"Transport {kind} does not exist")
            return
        del self.container.transports[kind]

    def replace_transport(self, new_transport: Transport):
        if new_transport.kind not in self.container.transports:
            logging.warning(f"Transport {new_transport.kind} does not exist")
        self.container.transports[new_transport.kind] = new_transport

    def set_character(self, character: Character) -> bool:
        if self.is_busy():
            return False
        self.container.character = character
        return True

    def set_voice(self, voice: Voice) -> bool:
        if self.is_busy():
            return False
        self.container.voice = voice
        return True

    def pause(self):
        if self.container.pause():
            return True
        else:
            logging.info(f"pause failed, human state is {state_str[self._state.get_state()]}")
            return False

    def speak(self, data: Data, force=False):
        return self.container.put_text_data(data, force)

    def run(self):
        if self._start:
            return
        runtime.submit_task(
            self.container.process_text_data_worker,
            task_info=TaskInfo(
                name=f"container.process_text_data_worker"
            )
        )
        runtime.submit_task(
            self.container.process_audio_data_worker,
            task_info=TaskInfo(
                name=f"container.process_audio_data_worker"
            )
        )
        runtime.submit_task(
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
    import logging
    import asyncio
    import time

    from langchain_openai import ChatOpenAI
    from engine.utils.data import Data
    from engine.config import ONE_API_LLM_MODEL
    from engine.human.voice.tts_ali import AliTTSWrapper
    from engine.human.character.agent import SimpleAgent
    from engine.utils import get_file_path
    from engine.config import DEFAULT_VOICE_PROCESSOR_CONFIG, DEFAULT_AVATAR_PROCESSOR_CONFIG, WAV2LIP_PLAYER_CONFIG
    from engine.transport import Transport, TransportWebRTC, TransportPyAudio
    from engine.human.avatar import wav2lip, AvatarProcessor
    from engine.human.voice import VoiceProcessor, AliTTSWrapper
    from engine.human.character.processor import BaseProcessor

    a_f = '../../../avatars/wav2lip256_avatar1'
    a_p = get_file_path(a_f)
    c_f = '../../../checkpoints/wav2lip/wav2lip.pth'
    c_p = get_file_path(c_f)

    # 创建Player实例并启动

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

    avatar_resource = wav2lip.load_avatar_resource(a_p.absolute().as_posix())
    avatar_model = wav2lip.Wav2LipWrapper(c_p.absolute().as_posix())

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
    character = Character(
        agent_model=agent,
        agent_processor=BaseProcessor(),  # 可以自定义处理器
    )

    # vector_store = try_load_db(DEFAULT_PROJECT_CONFIG.vecdb_path, DEFAULT_PROJECT_CONFIG.docs_path)
    # agent = KnowledgeAgent(llm_model, vector_store)

    voice_processor = VoiceProcessor(DEFAULT_VOICE_PROCESSOR_CONFIG)
    avatar_processor = AvatarProcessor(DEFAULT_AVATAR_PROCESSOR_CONFIG)

    webrtc_transport = TransportWebRTC(WAV2LIP_PLAYER_CONFIG)
    pyaudio_transport = TransportPyAudio(WAV2LIP_PLAYER_CONFIG)

    avatar = Avatar(
        avatar_resource=avatar_resource,
        avatar_model=avatar_model,
        avatar_processor=avatar_processor,
    )
    voice = Voice(
        tts_model=tts_model,
        voice_processor=voice_processor,
    )
    player = HumanPlayer(
        config=WAV2LIP_PLAYER_CONFIG,
        character=character,
        avatar=avatar,
        voice=voice,
        loop=runtime.main_loop,
        transports=[pyaudio_transport],
    )
    player.run()

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
            time.sleep(5)
            res_data = player.speak(Data(
                data="介绍故宫",
                is_chat=True,
                stream=True,
            ))
            logging.info(res_data)


    runtime.run_coroutine_threadsafe(listen_audio())
    runtime.run_coroutine_threadsafe(listen_video())
    runtime.run_coroutine_threadsafe(put_text_data())
    runtime.run_forever()
