import asyncio
import copy
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
from engine.human.player.track import StreamTrackSync
from engine.human.voice.voice import TTSModelWrapper
from engine.utils.data import Data


class HumanContainer:

    def __init__(
            self,
            config: PlayerConfig,
            agent: BaseAgent,
            tts_model: TTSModelWrapper,
            avatar_model: AvatarModelWrapper,
            avatar: Tuple,
            track_sync: StreamTrackSync,
            loop: asyncio.AbstractEventLoop,
    ):
        self.config = config
        self.agent = agent
        self.tts_model = tts_model
        self.avatar_model = avatar_model
        self.avatar = avatar
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        self.track_sync = track_sync
        self.loop = loop

        # runtime control
        self.state =  HumanState(StateReady)
        self.stop_event = threading.Event()

        # data flow
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # from config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.frame_multiple = config.frame_multiple
        self.timeout = config.timeout
        self.chunk_size = int(self.sample_rate / self.fps)
        self.batch_size = config.batch_size

        # temp
        self.audio_data_fragment = None
        self.audio_chunk_batch = []
        self.frame_index = 0
        self.frame_count = len(self.frame_list_cycle)

    def swap_state(self, old_state: int, new_state: int):
        res = self.state.swap_state(old_state, new_state)
        if res:
            print(f"swap: {state_str[old_state]} -> {state_str[new_state]}")
        return res

    def set_state(self, state: int):
        print(f"set:  {state_str[self.state.get_state()]} -> {state_str[state]}")
        self.state.set_state(state)

    def get_state(self):
         return self.state.get_state()

    def flush(self):
        # 如果非强制, 则只在 ready 状态下中止
        self.set_state(StatePause)
        self.text_queue.queue.clear()
        self.audio_queue.queue.clear()
        self.audio_chunk_batch = []
        self.audio_data_fragment = None
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
            if text is None or text == "":
                continue
            print(f"开始消费文本数据: {text}, {is_chat}")

            # todo, stream 支持
            stream = text_data.get("stream", False)
            try:
                if is_chat:
                    text = self.agent.answer(question=text)
                    print(f"Answer: {text}")

                speech = self.tts_model.inference(text)
                if speech is None:
                    print(f"Failed to answer: {text}")
                    continue
                is_final = True
                audio_data = Data(
                    data=speech,
                    is_final=is_final,
                )
                audio_data_chunks = self._split_audio_data_chunks(
                    audio_data
                )
                is_final_frame = False
                for i, chunk in enumerate(audio_data_chunks):
                    is_final_frame = is_final and i == len(audio_data_chunks)-1
                    self.audio_queue.put(
                        Data(
                            data=chunk,
                            is_final=is_final_frame,
                        )
                    )
                if is_final_frame:
                    print(f"Final frame")
            except Exception as e:
                print(f"Process text data error: {e}, text: {text}")
                traceback.print_exc()
                # 遇到错误, 状态重置为 ready
                self.set_state(StateReady)
                continue

    def _read_audio_frame(self):
        try:
            audio_data = self.audio_queue.get(timeout=self.timeout)
            chunk = audio_data.get("data")
            is_final = audio_data.get("is_final")
            state=1
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
        if self._mirror_frame_index(self.frame_index) == 0:
            self.frame_index = 0

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

    def process_audio_data_worker(self):
        while not self.stop_event.is_set():
            # 当前状态可能为 ready, busy 和 speaking, 状态为 pause, 在进行 flush 操作, 暂时不继续读取帧
            if self.get_state() == StatePause:
                time.sleep(self.timeout)
                continue
            is_final = False
            silence = True
            audio_frame_batch = []
            for i in range(self.batch_size * self.frame_multiple):
                audio_frame_data = self._read_audio_frame()
                audio_chunk = audio_frame_data.get("data")
                audio_frame_batch.append(self._make_audio_frame(audio_chunk))
                _is_final = audio_frame_data.get("is_final")
                _state = audio_frame_data.get("state")
                if not is_final:
                    is_final = _is_final
                if _state == 1:
                    silence = False
                chunk = audio_frame_data.get("data")
                self.audio_chunk_batch.append(chunk)
            try:
                audio_feature_batch = self.avatar_model.encode_audio_feature(self.audio_chunk_batch, self.config)
            except Exception as e:
                print(f"Encode audio feature error: {e}")
                traceback.print_exc()
                # 遇到错误, 状态重置为 ready
                self.set_state(StateReady)
                continue
            self.audio_chunk_batch = self.audio_chunk_batch[self.batch_size:]
            if silence:
                for i in range(self.batch_size):
                    frame_index = self._mirror_frame_index(self.frame_index)
                    video_frame = self.frame_list_cycle[frame_index]
                    video_frame = self._make_video_frame(video_frame)
                    asyncio.run_coroutine_threadsafe(self.track_sync.put_video_frame(video_frame), self.loop)
                    for j in range(self.frame_multiple):
                        audio_frame = audio_frame_batch[i * self.frame_multiple + j]
                        asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(audio_frame), self.loop)
                    self._update_frame_index(1)
            else:
                # 当前状态为 busy, 切换为 speaking
                self.swap_state(StateBusy, StateSpeaking)
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
                    print(f"Inference error: {e}")
                    traceback.print_exc()
                    # 遇到错误, 状态重置为 ready
                    self.set_state(StateReady)
                    continue

                for i, pred in enumerate(pred_img_batch):
                    frame_index = self._mirror_frame_index(self.frame_index)
                    video_frame = self._render_frame(pred, frame_index)
                    asyncio.run_coroutine_threadsafe(self.track_sync.put_video_frame(video_frame), self.loop)
                    for j in range(self.frame_multiple):
                        audio_frame = audio_frame_batch[i * self.frame_multiple + j]
                        asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(audio_frame), self.loop)
                    self._update_frame_index(1)

            if is_final:
                # 当前状态为 speaking, 最后一个帧, 状态切换回 ready
                self.swap_state(StateSpeaking, StateReady)

    def shutdown(self):
        self.stop_event.set()

