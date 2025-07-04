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
from torch import nn

from engine.agent.agents.base_agent import BaseAgent
from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG
from engine.human.avatar.avatar import AvatarModelWrapper
from engine.human.player.constant import *
from engine.human.player.state import HumanState
from engine.human.player.track import StreamTrackSync
from engine.utils.data import Data
from engine.human.voice.voice import TTSModelWrapper


class Container:
    state: HumanState
    stop_event = threading.Event()
    def __init__(self, state: HumanState):
        self.state = state

    def swap_state(self, old_state: int, new_state: int):
        return self.state.swap_state(old_state, new_state)

    def set_state(self, state: int):
        # print(f"{self.state.get_state()} -> {state}")
        self.state.set_state(state)

    def get_state(self):
         return self.state.get_state()

    def flush(self):
        pass

    def shutdown(self):
        self.stop_event.set()


class TextContainer(Container):
    def __init__(
            self,
            state: HumanState,
            config: PlayerConfig,
            agent: BaseAgent,
            model: TTSModelWrapper,
    ):
        """
        接收 text 数据, 如果是 chat 类型的文本, 先生成回答, 输出 audio
        :param config:
        :param agent:
        :param model:
        """
        super().__init__(state)

        self.config = config
        self.agent = agent
        self.model = model

        self.queue = queue.Queue()

    def put_text_data(self, data: Data) -> Data:
        # 强制进行后续操作
        if not self.swap_state(StateReady, StateBusy):
            return Data(
                ok=False,
                msg="busy"
            )

        # 文字预处理操作
        text = data.get("data")
        if text.startswith("fuck"):
            # 处理特殊文本（例如违法规定的文字）, 直接返回错误, 不放入队列
            return Data(
                ok=False,
                msg="fuck data",
            )

        self.queue.put(data)
        return Data(
            ok=True,
        )

    def audio_producer(self):
        """
        进一步处理 audio, 如变声, 增强等
        :return:
        """
        while not self.stop_event.is_set():
            try:
                text_data = self.queue.get(timeout=1)
            except queue.Empty:
                continue

            text = text_data.get("data")
            is_chat = text_data.get("is_chat", False)
            try:
                if text is None or text == "":
                    continue
                print(f"开始消费文本数据: {text}, {is_chat}")
                if is_chat:
                    text = self.agent.answer(question=text)

                print(f"Answer: {text}")
                speech_data = self.model.inference(text)
                yield Data(
                    data=speech_data,
                    is_final=True,
                )
                self.set_state(StateReady)
            except Exception as e:
                print(f"Text to audio error: {e}, text: {text}")
                traceback.print_exc()
                continue


class AudioContainer(Container):
    def __init__(
            self,
            state: HumanState,
            config: PlayerConfig,
            model: AvatarModelWrapper,
            track_sync: StreamTrackSync,
            loop: asyncio.AbstractEventLoop,
    ):
        """
        接收 audio 数据, 按照 fps 切割为 chunk; 组装 batch, 提取 audio feature
        :param config:
        :param model:
        """
        super().__init__(state)

        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.timeout = config.timeout
        self.chunk_size = int(self.sample_rate / self.fps)
        self.batch_size = config.batch_size
        self.model = model
        self.track_sync = track_sync
        self.loop = loop

        self.queue = queue.Queue(self.fps * 3)
        self.frame_queue = queue.Queue(self.fps * 3)
        self.frame_batch = []
        self.frame_fragment = None

    def flush(self):
        self.queue.queue.clear()
        self.frame_queue.queue.clear()
        self.frame_batch = []
        self._warmup()
        self.frame_fragment = None
        self.track_sync.flush()

    def audio_consumer(self, data: Data):
        self.flush()
        is_final = data.get("is_final")
        speech_data = data.get("data")
        if self.frame_fragment is not None:
            speech_data = np.concatenate([self.frame_fragment, speech_data])

        chunk_count = int((len(speech_data) - 1) / self.chunk_size) + 1
        for i in range(chunk_count):
            chunk = speech_data[i * self.chunk_size:(i + 1) * self.chunk_size]
            if not is_final and i == chunk_count - 1 and len(chunk) < self.chunk_size:
                self.frame_fragment = chunk
            else:
                self.queue.put(chunk)
        self.frame_fragment = None

    def _make_audio_frame(self, chunk):
        chunk = (chunk * 32767).astype(np.int16)
        frame = AudioFrame(format='s16', layout='mono', samples=chunk.shape[0])
        frame.planes[0].update(chunk.tobytes())
        frame.sample_rate = self.sample_rate
        return frame

    def _read_frame(self):
        try:
            chunk = self.queue.get(timeout=self.timeout)
            state=1
        except queue.Empty:
            chunk = np.zeros(self.chunk_size, dtype=np.float32)
            state = 0
        frame = self._make_audio_frame(chunk)
        data = Data(
            data=chunk,
            state=state,
        )
        return data, frame

    def _warmup(self):
        warmup_iters = max(self.config.warmup_iters, self.batch_size)
        for _ in range(warmup_iters):
            data, frame = self._read_frame()
            chunk = data.get("data")
            asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(frame), self.loop)
            self.frame_batch.append(chunk)

    def audio_feature_producer(self):
        self._warmup()
        while not self.stop_event.is_set():
            try:
                silence = True
                for i in range(self.batch_size):
                    data, frame = self._read_frame()
                    chunk, state = data.get("data"), data.get("state")
                    asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(frame), self.loop)
                    self.frame_batch.append(chunk)
                    if state == 1:
                        silence = False

                try:
                    audio_feature_batch = self.model.encode_audio_feature(self.frame_batch, self.config)
                except Exception as e:
                    print(f"Encode audio feature error: {e}")
                    traceback.print_exc()
                    continue

                self.frame_batch = self.frame_batch[self.batch_size:]
                yield Data(
                    data=audio_feature_batch,
                    silence=silence,
                )
                print(f"Produce audio feature")

            except Exception as e:
                print(f"Audio processing error: {e}")
                traceback.print_exc()
                break


class VideoContainer(Container):
    def __init__(
            self,
            state: HumanState,
            config: PlayerConfig,
            model: nn.Module,
            avatar: Tuple,
            track_sync: StreamTrackSync,
            loop: asyncio.AbstractEventLoop,
    ):
        """
        接收 audio feature, 生成视频帧
        :param config:
        :param model:
        :param avatar:
        :param track_sync:
        :param loop:
        """
        super().__init__(state)

        self.config = config
        self.batch_size = config.batch_size
        self.fps = config.fps
        self.model = model
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        self.track_sync = track_sync
        self.loop = loop

        self.frame_count = len(self.frame_list_cycle)
        self.frame_index = 0

    def _mirror_index(self, index):
        turn = index // self.frame_count
        res = index % self.frame_count
        if turn % 2 == 0:
            return res
        else:
            return self.frame_count - res - 1

    def _update_index(self, n):
        self.frame_index += n

    def _make_video_frame(self, image):
        image[0, :] &= 0xFE  # 确保第一行是偶数，避免某些视频问题
        frame = VideoFrame.from_ndarray(image, format="bgr24")
        return frame

    def audio_feature_consumer(self, data: Data):
        audio_feature_batch = data.get("data")
        silence = data.get("silence")
        if silence:
            for _ in range(self.batch_size):
                frame_index = self._mirror_index(self.frame_index)
                frame = self.frame_list_cycle[frame_index]
                frame = self._make_video_frame(frame)
                self._update_index(1)

                asyncio.run_coroutine_threadsafe(self.track_sync.put_video_frame(frame), self.loop)
        else:
            face_img_batch = []
            for i in range(self.batch_size):
                frame_index = self._mirror_index(self.frame_index + i)
                face_img = self.face_list_cycle[frame_index]
                face_img_batch.append(face_img)

            # 模型推理
            face_img_batch = np.asarray(face_img_batch)
            audio_feature_batch = np.asarray(audio_feature_batch)
            face_img_batch = torch.FloatTensor(face_img_batch).to(DEFAULT_RUNTIME_CONFIG.device)
            audio_feature_batch = torch.FloatTensor(audio_feature_batch).to(DEFAULT_RUNTIME_CONFIG.device)

            try:
                with torch.no_grad():
                    pred_img_batch = self.model.inference(audio_feature_batch, face_img_batch, self.config)
            except Exception as e:
                print(f"Inference error: {e}")
                traceback.print_exc()

            for i, pred in enumerate(pred_img_batch):
                frame_index = self._mirror_index(self.frame_index)
                frame = copy.deepcopy(self.frame_list_cycle[frame_index])
                bbox = self.coord_list_cycle[frame_index]
                y1, y2, x1, x2 = bbox
                pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = pred
                frame = self._make_video_frame(frame)
                self._update_index(1)

                asyncio.run_coroutine_threadsafe(self.track_sync.put_video_frame(frame), self.loop)


if __name__ == '__main__':
    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.runtime import thread_pool
    from engine.utils.concurrent.pool import TaskInfo

    f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr_example.wav'
    c_f = '../../../checkpoints/wav2lip.pth'
    model = Wav2LipWrapper(c_f)

    # 创建Player实例并启动
    loop = asyncio.new_event_loop()

    track_sync = StreamTrackSync(WAV2LIP_PLAYER_CONFIG)
    audio_c = AudioContainer(
        config=WAV2LIP_PLAYER_CONFIG,
        model=model,
        track_sync=track_sync,
        loop=loop,
    )
    video_c = VideoContainer(
        config=WAV2LIP_PLAYER_CONFIG,
        model=model,
        avatar=load_avatar(f),
        track_sync=track_sync,
        loop=loop,
    )

    from engine.human.voice import soundfile_producer
    from engine.utils.concurrent.pipeline import Pipeline

    pipeline_audio = Pipeline(
        producer=soundfile_producer(s_f, fps=10),
        consumer=audio_c.audio_consumer,
    )

    pipeline_video = Pipeline(
        producer=audio_c.audio_feature_producer,
        consumer=video_c.audio_feature_consumer,
    )

    for pipe in [pipeline_audio, pipeline_video]:
        thread_pool.submit(
            pipe.produce_worker,
            task_info=TaskInfo(
                name=f"produce_worker"
            )
        )
        thread_pool.submit(
            pipe.consume_worker,
            task_info=TaskInfo(
                name=f"consume_worker"
            )
        )


    asyncio.set_event_loop(loop)
    loop.run_forever()

    thread_pool.shutdown()
