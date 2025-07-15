import asyncio
import logging
import queue
import threading
import time
import traceback
from typing import Union, List, Tuple

import numpy as np

from engine.human.character import Character
from engine.config import PlayerConfig
from engine.human.avatar import Avatar
from engine.human.player.state import *
from engine.transport import Transport
from engine.human.voice import Voice
from engine.utils.data import Data
from engine.utils.concurrent import SharedFlag


class HumanContainer:

    def __init__(
            self,
            config: PlayerConfig,
            character: Character,
            voice: Voice,
            avatar: Avatar,
            loop: asyncio.AbstractEventLoop,
            transports: List[Transport]=None,
    ):
        self.config = config
        self.character = character
        self.voice = voice
        self.avatar = avatar
        self.loop = loop

        self.transports = {}
        self.audio_only = True
        for transport in transports:
            self.transports[transport.kind] = transport
            if self.audio_only:
                self.audio_only = transport.audio_only

        # from config
        self.fps = config.fps
        self.timeout = config.timeout
        self.frame_multiple = config.frame_multiple
        self.chunk_size = int(config.sample_rate / self.fps)
        self.batch_size = config.batch_size

        # runtime control
        self.state =  HumanState(StateReady)
        self.stop_event = threading.Event()

        # data flow
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.frame_queue = queue.Queue(self.fps // self.frame_multiple)

        # temp
        self.audio_data_fragment = None
        self.silence_flag = SharedFlag(1)  # 1 静音 0 发声

    def swap_state(self, old_state: int, new_state: int):
        res = self.state.swap_state(old_state, new_state)
        if res:
            logging.info(f"swap: {state_str[old_state]} -> {state_str[new_state]}")
        return res

    def set_state(self, state: int):
        logging.info(f"set:  {state_str[self.state.get_state()]} -> {state_str[state]}")
        self.state.set_state(state)

    def get_state(self):
         return self.state.get_state()

    def add_transport(self, transport: Transport):
        if transport.kind in self.transports:
            logging.warning(f"Transport {transport.kind} already exists")
            return
        self.transports[transport.kind] = transport

    def remove_transport(self, kind: str):
        if kind not in self.transports:
            logging.warning(f"Transport {kind} does not exist")
            return
        del self.transports[kind]

    def replace_transport(self, new_transport: Transport):
        if new_transport.kind not in self.transports:
            logging.warning(f"Transport {new_transport.kind} does not exist")
        self.transports[new_transport.kind] = new_transport

    def pause(self):
        # 中断数字人当前对话
        if self.swap_state(StateSpeaking, StatePause) or self.swap_state(StateBusy, StatePause):
            self.text_queue.queue.clear()
            self.audio_queue.queue.clear()
            self.audio_data_fragment = None
        else:
            logging.info(f"pause failed, human state is {state_str[self.get_state()]}")

    def _need_resume(self):
        if self.get_state() == StatePause and self.silence_flag.get() == 1:
            return True
        else:
            return False

    def put_text_data(self, data: Data, force=False):
        if force:
            self.set_state(StateReady)
        if self._need_resume():
            self.swap_state(StatePause, StateReady)
        if not self.swap_state(StateReady, StateBusy):
            return Data(
                ok=False,
                msg=f"human state not ready, state is {state_str[self.get_state()]}",
            )

        # 文字预处理操作
        if not self.character.precheck(data.get("data")):
            return Data(
                ok=False,
                msg=f"Character check failed, invalid input text: {str(data)}",
            )

        self.text_queue.put(data)
        return Data(
            ok=True,
        )

    def _split_audio_data_chunks(self, data: Data):
        audio_data = data.get("data")
        is_final = data.get("is_final")
        if audio_data is None:
            return []
        if self.audio_data_fragment is not None:
            audio_data = np.concatenate([self.audio_data_fragment, audio_data])
            self.audio_data_fragment = None

        chunk_count = int((len(audio_data) - 1) / self.chunk_size) + 1
        audio_data_chunks = []
        for i in range(chunk_count):
            chunk = audio_data[i * self.chunk_size:(i + 1) * self.chunk_size]
            if not is_final and i == chunk_count - 1 and len(chunk) < self.chunk_size:
                self.audio_data_fragment = chunk
            else:
                audio_data_chunks.append(chunk)
        return audio_data_chunks

    def _produce_audio_data(self, speech: np.ndarray):
        if self.get_state() == StatePause:
            return
        if speech is None:
            return
        audio_data = Data(
            data=speech,
            is_final=False,
        )
        audio_data_chunks = self._split_audio_data_chunks(
            audio_data
        )
        for i, chunk in enumerate(audio_data_chunks):
            self.audio_queue.put(
                Data(
                    data=chunk,
                    is_final=False,
                )
            )

    def _streaming_answer_generator(self, text: str):
        for answer in self.character.stream_answer(question=text):
            yield answer, self.get_state() == StatePause
        yield "", True

    def process_text_data_worker(self):
        """
        text -> text answer -> audio -> audio chunks
        :return:
        """
        while not self.stop_event.is_set():

            try:
                text_data = self.text_queue.get(timeout=1)
            except queue.Empty:
                continue
            text = text_data.get("data")
            is_chat = text_data.get("is_chat", False)
            stream = text_data.get("stream")
            logging.info(f"开始消费文本数据: {text}, is_chat={is_chat}, stream={stream}")
            try:
                if not is_chat:
                    speech = self.voice.speak(text)
                    self._produce_audio_data(speech)
                else:
                    if stream:
                        self.voice.realtime_speak(
                            generator=self._streaming_answer_generator(text),
                            receiver=self._produce_audio_data,
                        )
                    else:
                        text = self.character.answer(question=text)
                        speech = self.voice.speak(text)
                        self._produce_audio_data(speech)
                self.audio_queue.put(
                    Data(
                        data=None,
                        is_final=True
                    )
                )
            except Exception as e:
                logging.info(f"Process text data error: {e}, text: {text_data.get('data')}")
                traceback.print_exc()
                # 遇到错误, 状态重置为 ready
                self.set_state(StateReady)
                continue

    def _read_audio_frame(self):
        try:
            audio_data = self.audio_queue.get(timeout=self.timeout)
            chunk = audio_data.get("data")
            state=1
            if chunk is None:
                chunk = np.zeros(self.chunk_size, dtype=np.float32)
                state = 0
            if self.get_state() == StatePause:
                chunk = np.zeros(self.chunk_size, dtype=np.float32)
                state = 0
            is_final = audio_data.get("is_final")
        except queue.Empty:
            chunk = np.zeros(self.chunk_size, dtype=np.float32)
            is_final = False
            state = 0
        data = Data(
            data=chunk,
            state=state,
            is_final=is_final,
        )
        return data

    def process_audio_data_worker(self):
        while not self.stop_event.is_set():
            try:
                is_final = False
                silence = True
                audio_chunk_batch = []
                for i in range(self.batch_size * self.frame_multiple):
                    audio_frame_data = self._read_audio_frame()
                    _is_final = audio_frame_data.get("is_final")
                    _state = audio_frame_data.get("state")
                    _audio_chunk = audio_frame_data.get("data")

                    if not is_final:
                        is_final = _is_final
                    if _state == 1:
                        silence = False
                    audio_chunk_batch.append(_audio_chunk)

                # process frames
                silence_flag = 1 if silence else 0
                self.silence_flag.set(silence_flag)
                if self.audio_only:
                    for i in range(self.batch_size):
                        audio_frames = audio_chunk_batch[
                                       i * self.frame_multiple:
                                       i * self.frame_multiple + self.frame_multiple]
                        self.frame_queue.put((None, audio_frames))
                else:
                    if silence:
                        i = 0
                        for video_frame in self.avatar.silence(self.config):
                            audio_frames = audio_chunk_batch[
                                           i * self.frame_multiple:
                                           i * self.frame_multiple + self.frame_multiple]
                            self.frame_queue.put((video_frame, audio_frames))
                            i += 1
                    else:
                        # 当前状态为 busy, 切换为 speaking
                        self.swap_state(StateBusy, StateSpeaking)
                        i = 0
                        for video_frame in self.avatar.speak(audio_chunk_batch, self.config):
                            audio_frames = audio_chunk_batch[
                                           i * self.frame_multiple:
                                           i * self.frame_multiple + self.frame_multiple]
                            self.frame_queue.put((video_frame, audio_frames))
                            i += 1

                if is_final:
                    self.set_state(StateReady)
            except Exception as e:
                logging.info(f"Process audio data error: {e}")
                traceback.print_exc()
                self.set_state(StateReady)
                time.sleep(self.timeout)
                continue

    def process_frames_worker(self):
        while not self.stop_event.is_set():
            try:
                video_frame, audio_frames = self.frame_queue.get(timeout=self.timeout)
            except queue.Empty:
                continue

            for transport in self.transports.values():
                if not transport.audio_only:
                    res = asyncio.run_coroutine_threadsafe(transport.put_video_frame(video_frame), self.loop)
                    if transport.wait:  # 避免由于 frame 生产速率过快时, 导致帧跳现象(频繁卡顿或是帧过快)
                        res.result()
                for audio_frame in audio_frames:
                    audio_frame = (audio_frame * 32767).astype(np.int16)  # to pa_int16
                    res = asyncio.run_coroutine_threadsafe(transport.put_audio_frame(audio_frame), self.loop)
                    if transport.wait:  # 避免由于 frame 生产速率过快时, 导致帧跳现象(频繁卡顿或是帧过快)
                        res.result()

    def shutdown(self):
        self.stop_event.set()
        self.set_state(StateNotReady)

