import asyncio
import copy
import logging
import queue
import threading
import time
import traceback
from typing import Tuple

import cv2
import numpy as np
import torch
from av import AudioFrame, VideoFrame

from engine.agent.agents.base_agent import BaseAgent
from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG
from engine.human.avatar.avatar import AvatarModelWrapper
from engine.human.player.state import *
from engine.human.player.track import StreamTrackSync, AudioStreamTrack, VideoStreamTrack
from engine.human.voice.voice import TTSModelWrapper
from engine.utils.data import Data


class HumanContainer:

    def __init__(
            self,
            config: PlayerConfig,
            agent: BaseAgent,
            tts_model: TTSModelWrapper,
            avatar_model: AvatarModelWrapper,
            loop: asyncio.AbstractEventLoop,
    ):
        self.config = config
        self.agent = agent
        self.tts_model = tts_model
        self.avatar_model = avatar_model
        self.loop = loop

        # track
        self.track_sync = StreamTrackSync(
            self.config,
        )
        self.audio_track = AudioStreamTrack(
            self.config,
            self.track_sync,
        )
        self.video_track = VideoStreamTrack(
            self.config,
            self.track_sync,
        )

        # avatar
        self.avatar: Tuple = None
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = None, None, None

        # runtime control
        self.state =  HumanState(StateNotReady)
        self.stop_event = threading.Event()

        # data flow
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # from config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.timeout = config.timeout
        self.audio_ptime = config.audio_ptime
        self.video_ptime = config.video_ptime
        self.chunk_size = int(self.sample_rate / self.fps)
        self.batch_size = config.batch_size

        # temp
        self.audio_data_fragment = None
        self.audio_data_count = 0
        self.audio_chunk_batch = []
        self.audio_frame_batch = []
        self.frame_index = 0
        self.frame_count = 0

    def load(self):
        self.avatar = self.avatar_model.load_avatar()
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = self.avatar
        self.frame_count = len(self.frame_list_cycle)
        self.set_state(StateReady)

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
        self.set_state(StatePause)
        self.text_queue.queue.clear()
        self.audio_queue.queue.clear()
        self.audio_chunk_batch = []
        self.audio_data_fragment = None
        # 完成 flush 操作后, 切换到 ready 状态
        self.set_state(StateReady)

    def flush_track(self):
        self.frame_index = self.track_sync.flush()

    def flush(self):
        # 中断数字人当前对话
        self.set_state(StatePause)
        self.text_queue.queue.clear()
        self.audio_queue.queue.clear()
        self.audio_chunk_batch = []
        self.audio_data_fragment = None
        self.flush_track()
        # 完成 flush 操作后, 切换到 ready 状态
        self.set_state(StateReady)

    def put_text_data(self, data: Data):
        # human busy
        if self.get_state() != StateReady and self.get_state() != StateSpeaking:
            return Data(
                ok=False,
                msg=f"human state is {state_str[self.get_state()]}",
            )
        self.set_state(StateBusy)

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
                    else:
                        logging.info("final frame")
                        self.tts_model.complete()
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

    def _mirror_frame_index(self, index):
        turn = index // self.frame_count
        res = index % self.frame_count
        if turn % 2 == 0:
            return res
        else:
            return self.frame_count - res - 1

    def _update_frame_index(self, n):
        self.frame_index += n

    def _make_audio_frame(self, chunk):
        chunk = (chunk * 32767).astype(np.int16)
        frame = AudioFrame(format='s16', layout='mono', samples=chunk.shape[0])
        frame.planes[0].update(chunk.tobytes())
        frame.sample_rate = self.sample_rate
        return frame

    def _make_video_frame(self, image):
        image[0, :] &= 0xFE  # 确保第一行是偶数，避免某些视频问题
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        return frame

    def _render_frame(self, pred, frame_index):
        frame = copy.deepcopy(self.frame_list_cycle[frame_index])
        bbox = self.coord_list_cycle[frame_index]
        y1, y2, x1, x2 = bbox
        pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))
        frame[y1:y2, x1:x2] = pred
        frame = self._make_video_frame(frame)
        return frame

    def _warmup(self):
        for i in range(20):
            audio_frame_data = self._read_audio_frame()
            audio_chunk = audio_frame_data.get("data")
            self.audio_chunk_batch.append(audio_chunk)
            self.audio_frame_batch.append(self._make_audio_frame(audio_chunk))
        for _ in range(10):
            self.audio_frame_batch.pop(0)

    def process_audio_data_worker(self):
        self._warmup()
        while not self.stop_event.is_set():
            # 当前状态可能为 ready, busy 和 speaking, 状态为 pause, 在进行 flush 操作, 暂时不继续读取帧
            if self.get_state() == StatePause:
                time.sleep(self.timeout)
                continue
            is_final = False
            silence = True
            for i in range(self.batch_size*2):
                audio_frame_data = self._read_audio_frame()
                audio_chunk = audio_frame_data.get("data")
                self.audio_frame_batch.append(self._make_audio_frame(audio_chunk))
                _is_final = audio_frame_data.get("is_final")
                _state = audio_frame_data.get("state")
                if not is_final:
                    is_final = _is_final
                if _state == 1:
                    silence = False
                chunk = audio_frame_data.get("data")
                self.audio_chunk_batch.append(chunk)
            if not silence:
                try:
                    audio_feature_batch = self.avatar_model.encode_audio_feature(self.audio_chunk_batch, self.config)
                except Exception as e:
                    logging.info(f"Encode audio feature error: {e}")
                    traceback.print_exc()
                    # 遇到错误, 状态重置为 ready
                    self.set_state(StateReady)
                    continue
            else:
                audio_feature_batch = None
            self.audio_chunk_batch = self.audio_chunk_batch[self.batch_size*2:]
            audio_frame_batch = self.audio_frame_batch[:self.batch_size*2]
            self.audio_frame_batch = self.audio_frame_batch[self.batch_size*2:]
            if silence:
                for i in range(self.batch_size):
                    frame_index = self._mirror_frame_index(self.frame_index)
                    video_frame = self.frame_list_cycle[frame_index]
                    video_frame = self._make_video_frame(video_frame)
                    asyncio.run_coroutine_threadsafe(self.track_sync.put_video_frame(video_frame), self.loop)
                    audio_frames = audio_frame_batch[i*2:i*2+2]
                    for audio_frame in audio_frames:
                        asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(audio_frame), self.loop)
                    self._update_frame_index(1)
            else:
                # 当前状态为 busy, 切换为 speaking
                if self.swap_state(StateBusy, StateSpeaking):
                    self.flush_track()
                face_img_batch = []
                for i in range(self.batch_size):
                    frame_index = self._mirror_frame_index(self.frame_index + i)
                    face_img = self.face_list_cycle[frame_index]
                    face_img_batch.append(face_img)

                face_img_batch = np.asarray(face_img_batch)
                audio_feature_batch = np.asarray(audio_feature_batch)
                face_img_batch = torch.FloatTensor(face_img_batch).to(DEFAULT_RUNTIME_CONFIG.device)
                audio_feature_batch = torch.FloatTensor(audio_feature_batch).to(DEFAULT_RUNTIME_CONFIG.device)

                try:
                    with torch.no_grad():
                        pred_img_batch = self.avatar_model.inference(audio_feature_batch, face_img_batch, self.config)
                except Exception as e:
                    logging.info(f"Inference error: {e}")
                    traceback.print_exc()
                    # 遇到错误, 状态重置为 ready
                    self.set_state(StateReady)
                    continue

                for i, pred in enumerate(pred_img_batch):
                    frame_index = self._mirror_frame_index(self.frame_index)
                    video_frame = self._render_frame(pred, frame_index)
                    asyncio.run_coroutine_threadsafe(self.track_sync.put_video_frame(video_frame), self.loop)
                    audio_frames = audio_frame_batch[i*2:i*2+2]
                    for audio_frame in audio_frames:
                        asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(audio_frame), self.loop)
                    self._update_frame_index(1)

            if is_final:
                # 当前状态为 speaking, 最后一个帧, 状态切换回 ready
                self.swap_state(StateSpeaking, StateReady)

    def shutdown(self):
        self.stop_event.set()
        self.set_state(StateNotReady)

