import asyncio
import logging
import queue
import threading
import time
import traceback
from typing import Union, List, Tuple

import numpy as np
import torch

from engine.human.character.agent.base_agent import BaseAgent
from engine.config import PlayerConfig
from engine.human.avatar import AvatarModelWrapper, Avatar, AvatarProcessor
from engine.human.player.state import *
from engine.transport import Transport
from engine.human.voice import TTSModelWrapper, VoiceProcessor
from engine.utils.data import Data
from engine.utils.concurrent import SharedFlag


class HumanContainer:

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
        self.agent = agent
        self.tts_model = tts_model
        self.avatar_model = avatar_model
        self.voice_processor = voice_processor
        self.avatar_processor = avatar_processor
        self.loop = loop
        self.main_transport = main_transport
        if other_transports is not None:
            if isinstance(other_transports, Transport):
                self.other_transports = [other_transports]
            else:
                self.other_transports = list(other_transports)
        else:
            self.other_transports = []

        # from config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.timeout = config.timeout
        self.audio_ptime = config.audio_ptime
        self.video_ptime = config.video_ptime
        self.frame_multiple = config.frame_multiple
        self.chunk_size = int(self.sample_rate / self.fps)
        self.batch_size = config.batch_size

        # avatar
        self.avatar: Avatar = avatar

        # runtime control
        self.state =  HumanState(StateReady)
        self.stop_event = threading.Event()

        # data flow
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.frame_queue = queue.Queue(self.fps // self.frame_multiple)

        # temp
        self.audio_data_fragment = None
        self.audio_data_count = 0
        self.audio_chunk_batch = []

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
        text = data.get("data")
        if text.startswith("fuck"):
            # todo, 预处理和过滤（例如违法规定的文字）, 直接返回错误, 不放入队列
            return Data(
                ok=False,
                msg="fuck data",
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

    def _generate_answer_data(self, text_data: Data):
        text = text_data.get("data")
        is_chat = text_data.get("is_chat", False)
        stream = text_data.get("stream")
        logging.info(f"开始消费文本数据: {text}, is_chat={is_chat}, stream={stream}")

        final_data = Data(
            data="",
            is_final=True,
        )
        echo_data = Data(
            data=text,
            is_final=False,
        )
        if text is None or text == "":
            yield final_data
            return
        if not is_chat:
            yield echo_data  # todo echo data 做拆分
            yield final_data
            return

        if stream:
            for answer in self.agent.stream_answer(question=text):
                answer_data = Data(
                    data=answer,
                    is_final=False,
                )
                yield answer_data
            yield final_data
        else:
            answer_data = Data(
                data=self.agent.answer(question=text),
                is_final=False,
            )
            yield answer_data
            yield final_data

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
            self.audio_data_count += 1
            self.audio_queue.put(
                Data(
                    data=chunk,
                    is_final=False,
                )
            )

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
            stream = text_data.get("stream", False)
            self.audio_data_count = 0
            try:
                self.tts_model.reset(self._produce_audio_data)
                for answer_data in self._generate_answer_data(text_data):
                    if self.get_state() == StatePause:
                        break
                    answer = answer_data.get("data")
                    is_final = answer_data.get("is_final")
                    if len(answer) > 0 and not is_final:
                        # todo, 判断 answer是否仅为标点符号, 以及做一定的组装后再调用 tts model
                        logging.info(f"Answer: {answer}, is_final={is_final}")
                        if stream:
                            self.tts_model.streaming_inference(answer)
                        else:
                            speech = self.tts_model.inference(answer)
                            self._produce_audio_data(speech)
                    elif len(answer) == 0 and not is_final:
                        continue
                self.tts_model.complete()
                logging.info("final frame")
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

    def _process_audio_frame(self, chunk):
        chunk = self.voice_processor.process(chunk)
        chunk = (chunk * 32767).astype(np.int16)  # to pcm
        return chunk

    def _process_video_frame(self, image):
        image = self.avatar_processor.process(image)
        return image

    def process_audio_data_worker(self):
        while not self.stop_event.is_set():
            try:
                is_final = False
                silence = True
                audio_frame_batch = []

                # prepare audio data
                for i in range(self.batch_size * self.frame_multiple):
                    audio_frame_data = self._read_audio_frame()

                    _is_final = audio_frame_data.get("is_final")
                    _state = audio_frame_data.get("state")
                    if not is_final:
                        is_final = _is_final
                    if _state == 1:
                        silence = False

                    audio_chunk = audio_frame_data.get("data")
                    audio_frame_batch.append(self._process_audio_frame(audio_chunk))
                    self.audio_chunk_batch.append(audio_chunk)

                # process frames
                silence_flag = 1 if silence else 0
                self.silence_flag.set(silence_flag)
                if silence:
                    for i in range(self.batch_size):
                        frame = self.avatar.get_next_frame()
                        video_frame = self._process_video_frame(frame)
                        audio_frames = audio_frame_batch[
                                       i * self.frame_multiple:
                                       i * self.frame_multiple + self.frame_multiple]
                        self.frame_queue.put(
                            (video_frame, audio_frames)
                        )
                else:
                    # 当前状态为 busy, 切换为 speaking
                    self.swap_state(StateBusy, StateSpeaking)
                    with torch.no_grad():
                        pred_img_batch = self.avatar_model.inference(self.audio_chunk_batch, self.config)

                    for i, pred in enumerate(pred_img_batch):
                        frame = self.avatar.render_frame(pred)
                        video_frame = self._process_video_frame(frame)
                        audio_frames = audio_frame_batch[
                                       i * self.frame_multiple:
                                       i * self.frame_multiple + self.frame_multiple]
                        self.frame_queue.put(
                            (video_frame, audio_frames)
                        )

                self.audio_chunk_batch = []
                if is_final:
                    self.set_state(StateReady)
            except Exception as e:
                logging.info(f"Process audio data error: {e}")
                traceback.print_exc()
                self.set_state(StateReady)
                time.sleep(self.timeout)
                continue

    def _send_frames(self, transport, video_frame, audio_frames, main=False):
        res = asyncio.run_coroutine_threadsafe(transport.put_video_frame(video_frame), self.loop)
        if main:  # 适用于 webrtc, 避免由于 frame 生产速率过快时, track 为了对齐时间戳频繁 wait(不精确), 导致帧跳现象(频繁卡顿或是帧过快)
            res.result()
        for audio_frame in audio_frames:
            res = asyncio.run_coroutine_threadsafe(transport.put_audio_frame(audio_frame), self.loop)
            if main:
                res.result()

    def process_frames_worker(self):
        while not self.stop_event.is_set():
            try:
                video_frame, audio_frames = self.frame_queue.get(timeout=self.timeout)
            except queue.Empty:
                continue

            self._send_frames(self.main_transport, video_frame, audio_frames, main=True)
            for transport in self.other_transports:
                self._send_frames(transport, video_frame, audio_frames, main=False)

    def shutdown(self):
        self.stop_event.set()
        self.set_state(StateNotReady)

