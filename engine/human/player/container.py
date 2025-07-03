import asyncio
import copy
import queue
import threading
import traceback
from typing import Tuple

import cv2
import numpy as np
import torch
from av import AudioFrame, VideoFrame
from openai import AsyncOpenAI
from torch import nn
from typing_inspection.typing_objects import is_final

from engine.agent.agents.base_agent import BaseAgent
from engine.agent.agents.custom.agents import KnowledgeAgent
from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG
from engine.human.avatar.avatar import AvatarModelWrapper
from engine.human.player.track import StreamTrackSync
from engine.utils.data import Data
from engine.human.voice.voice import TTSModelWrapper


class Container:
    def flush(self):
        pass

    def shutdown(self):
        pass


class TextContainer(Container):
    def __init__(
            self,
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
        self.config = config
        self.agent = agent
        self.model = model

        self._stop_event = threading.Event()

        self.queue = queue.Queue()
        self.audio_queue = queue.Queue()

    def flush(self):
        self.queue.queue.clear()
        self.audio_queue.queue.clear()

    def put_text_data(self, data: Data) -> Data:
        text = data.get("data")
        is_chat = data.get("is_chat")
        try:
            if is_chat:
                # 将data放入agent并进行回答
                loop = asyncio.get_event_loop()
                async def handle_async():
                    async for chunk in await self.agent.stream_answer(text):
                        print(chunk)
                        self.queue.put(Data(data=chunk, is_chat=True))
                    self.queue.put(Data(is_final=True))

                loop.run_until_complete(handle_async())
            else:
                # 普通文本, 直接放入队列
                self.queue.put(Data(data=text, is_chat=False, is_final=True))
        except Exception as e:
            print(f"Error in agent processing: {e}")
            return Data(
                ok=False,
                msg=str(e),
            )
        return Data(ok=True)

    def text_producer(self):
        """
        读取 text, 如果 chat 类型, 则通过 agent 产生流式输出
        :return:
        """
        while not self._stop_event.is_set():
            try:
                text_data = self.queue.get(timeout=1)
                text = text_data.get("data")
                is_chat = text_data.get("is_chat", False)
                if not is_chat:
                    # echo
                    yield Data(
                        data=text,
                        is_stream=False,
                        is_final=True,
                    )
                else:
                    # chat
                    is_final = text_data.get("is_final", False)
                    if is_final:
                        # 最后发送一个标记数据, 代表是最后一个
                        yield Data(
                            data="",
                            is_stream=True,
                            is_final=True,
                        )
                    yield Data(
                        data=text,
                        is_stream=True,
                        is_final=False,
                    )
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Text processing error: {e}")
                traceback.print_exc()
                break


    def text_consumer(self, data: Data):
        """
         将文本或文本流, 转为音频
         :return:
         """
        text = data.get("data")
        is_stream = data.get("is_stream")
        is_final = data.get("is_final")
        if is_final and len(text) == 0:
            return
        if is_stream:
            speech_data = self.model.streaming_inference(text)
        else:
            speech_data = self.model.inference(text)
        audio_data = Data(
            data=speech_data,
            is_final=is_final,
        )
        self.audio_queue.put(audio_data)

    def audio_producer(self):
        """
        进一步处理 audio, 如变声, 增强等
        :return:
        """
        while not self._stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=1)
                print(audio_data)
                # todo @zjh, 进一步处理音频

                yield audio_data  # data, is_final
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
                traceback.print_exc()
                break

    def shutdown(self):
        self._stop_event.set()


class AudioContainer(Container):
    def __init__(
            self,
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
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.timeout = config.timeout
        self.chunk_size = int(self.sample_rate / self.fps)
        self.batch_size = config.batch_size
        self.model = model
        self.track_sync = track_sync
        self.loop = loop

        self._stop_event = threading.Event()

        self.queue = queue.Queue(self.fps * 3)
        self.frame_queue = queue.Queue(self.fps * 3)
        self.frame_batch = []
        self.frame_fragment = None

    def flush(self):
        self.queue.queue.clear()
        self.frame_queue.queue.clear()
        self.frame_batch = []
        self.frame_fragment = None

    def audio_consumer(self, data: Data):
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

    def _make_audio_frame(self, chunk):
        chunk = (chunk * 32767).astype(np.int16)
        frame = AudioFrame(format='s16', layout='mono', samples=chunk.shape[0])
        frame.planes[0].update(chunk.tobytes())
        frame.sample_rate = self.sample_rate
        return frame

    def _read_frame(self):
        try:
            chunk = self.queue.get(timeout=self.timeout)
            state = 1
        except queue.Empty:
            chunk = np.zeros(self.chunk_size, dtype=np.float32)
            state = 0

        frame = self._make_audio_frame(chunk)
        return chunk, state, frame

    def _warmup(self):
        warmup_iters = max(self.config.warmup_iters, self.batch_size)
        for _ in range(warmup_iters):
            chunk, state, frame = self._read_frame()
            asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(frame), self.loop)
            self.frame_batch.append(chunk)

    def audio_feature_producer(self):
        self._warmup()
        while not self._stop_event.is_set():
            try:
                silence = True
                for i in range(self.batch_size):
                    chunk, state, frame = self._read_frame()
                    asyncio.run_coroutine_threadsafe(self.track_sync.put_audio_frame(frame), self.loop)
                    self.frame_batch.append(chunk)
                    if state == 1:
                        silence = False
                audio_feature_batch = self.model.encode_audio_feature(self.frame_batch, self.config)
                self.frame_batch = self.frame_batch[self.batch_size:]
                yield Data(
                    data=audio_feature_batch,
                    silence=silence,
                )

            except Exception as e:
                print(f"Audio processing error: {e}")
                traceback.print_exc()
                break

    def shutdown(self):
        self._stop_event.set()


class VideoContainer(Container):
    def __init__(
            self,
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
    from engine.utils.pool import TaskInfo

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
    from engine.utils.pipeline import Pipeline

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
