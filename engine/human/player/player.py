import asyncio
import copy
from queue import Queue
from typing import Optional, Tuple, List
import queue

import numpy as np
import torch
from PIL import Image
from av import AudioFrame, VideoFrame
import cv2
from torch import nn
import torch.multiprocessing as mp

import pyaudio
from threading import Thread, Event
from queue import Queue

from engine.human.avatar.wav2lip import load_avatar, load_model
from engine.human.utils.data import Data
from engine.human.voice.asr import soundfile_producer
from engine.utils.pipeline import AsyncPipeline, AsyncConsumer, AsyncConsumerFactory, AsyncPipelineRunner
from models.wav2lip.audio import melspectrogram
from models.wav2lip.models import Wav2Lip

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEL_STEP_SIZE = 16
MEL_IDX_MULTIPLIER = 80. * 2
SAMPLE_RATE = 16000



class MediaPlayer:
    def __init__(self, fps=30, audio_format=pyaudio.paInt16,
                 channels=1, rate=44100):
        """
        初始化媒体播放器

        参数:
            fps: 视频帧率
            audio_format: 音频格式
            channels: 音频通道数
            rate: 音频采样率
        """
        self.fps = fps
        self.audio_format = audio_format
        self.channels = channels
        self.rate = rate

        # 创建视频帧和音频帧队列
        self.video_queue = Queue(maxsize=100)
        self.audio_queue = Queue(maxsize=100)

        # 事件控制
        self.stop_event = Event()

        # 初始化音频流
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(
            format=audio_format,
            channels=channels,
            rate=rate,
            output=True
        )

        # 启动播放线程
        self.video_thread = Thread(target=self._play_video)
        self.audio_thread = Thread(target=self._play_audio)

        self.video_thread.start()
        self.audio_thread.start()

    def add_video_frame(self, frame: np.ndarray):
        """添加视频帧到播放队列"""
        if not self.stop_event.is_set():
            try:
                # 如果队列已满，丢弃最旧的帧
                if self.video_queue.full():
                    self.video_queue.get_nowait()
                self.video_queue.put_nowait(frame)
            except:
                pass

    def add_audio_frame(self, audio_data: np.ndarray):
        """添加音频帧到播放队列"""
        if not self.stop_event.is_set():
            try:
                # 如果队列已满，丢弃最旧的帧
                if self.audio_queue.full():
                    self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except:
                pass

    def _play_video(self):
        """视频播放线程"""

        while not self.stop_event.is_set():
            try:
                # 获取视频帧，超时时间为1秒
                frame = self.video_queue.get(timeout=1)

                # 显示视频帧
                im = Image.fromarray(frame)
                im.show()
                im.close()

            except:
                pass

    def _play_audio(self):
        """音频播放线程"""
        while not self.stop_event.is_set():
            try:
                # 获取音频帧，超时时间为1秒
                audio_data = self.audio_queue.get(timeout=1)

                # 播放音频
                self.audio_stream.write(audio_data.tobytes())

            except:
                pass

    def stop(self):
        """停止播放并清理资源"""
        self.stop_event.set()

        # 等待线程结束
        if self.video_thread.is_alive():
            self.video_thread.join()

        if self.audio_thread.is_alive():
            self.audio_thread.join()

        # 关闭音频流
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.p.terminate()

        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()



# 初始化播放器
media_player = MediaPlayer(fps=50, rate=16000)


def encode(speech_frame_batch, fps=50):
    l = r = 10
    frames = np.concatenate(speech_frame_batch)
    mel = melspectrogram(frames)
    left = max(0, int(l * 80 / fps))
    right = min(len(mel[0]), len(mel[0]) - int(r * 80 / fps))
    mel_step_size = MEL_STEP_SIZE
    mel_idx_multiplier = int(MEL_IDX_MULTIPLIER / fps)
    i = 0
    mel_chunks = []
    while i < (len(speech_frame_batch) - l - r) / 2:
        start_idx = int(left + i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
        else:
            mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    speech_frame_batch = speech_frame_batch[-(l+r):]
    return mel_chunks, speech_frame_batch

class Player:
    def __init__(
            self,
            model: nn.Module,
            # model.load_state_dict, model.warmup, model.inference
            avatar: Tuple,
            fps: int,
            sample_rate: int,
            batch_size: int,
            stop_event: asyncio.Event,
            timeout_data: float,
            timeout_feature: float,
            timeout_frame: float,
    ):
        self.model = model
        self.fps = fps
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.chunk_size = int(self.sample_rate / self.fps)
        self.stop_event = stop_event
        self.timeout_data = timeout_data
        self.timeout_feature = timeout_feature
        self.timeout_frame = timeout_frame

        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar

        self.speech_frames = asyncio.Queue()
        self.speech_frames_2 = asyncio.Queue()
        self.speech_features = asyncio.Queue()
        self.res_frames = asyncio.Queue()  # avatar_frame, speech_frame, frame_idx

        # state
        self.speech_frame_batch = []
        self.frame_length = len(self.frame_list_cycle)
        self.frame_index = 0

    def mirror_index(self, index):
        turn = index // self.frame_length
        res = index % self.frame_length
        if turn % 2 == 0:
            return res
        else:
            return self.frame_length - res - 1

    def update_index(self, n):
        self.frame_index += n

    def make_avatar_frame(self, frame_idx, pred_frame=None):
        new_frame = copy.deepcopy(self.frame_list_cycle[frame_idx])
        if pred_frame is not None:
            bbox = self.coord_list_cycle[frame_idx]
            y1, y2, x1, x2 = bbox
            pred_frame = cv2.resize(pred_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            new_frame[y1:y2, x1:x2] = pred_frame
        return new_frame

    async def put_res_frame(self, avatar_frame, speech_frame, frame_idx):
        await self.res_frames.put((avatar_frame, speech_frame, frame_idx))

    # pipeline1
    def speech_data_consumer(self) -> AsyncConsumer:
        async def consume_fn(data: Data, processed_data: Data = None):
            await self.speech_frames.put(data)
        handler = None
        return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)

    def speech_data_producer(self):
        async def producer():
            while not self.stop_event.is_set() or not self.speech_frames.empty():
                try:
                    speech_chunk_data = await asyncio.wait_for(self.speech_frames.get(), timeout=self.timeout_data)
                    speech_chunk = speech_chunk_data.get('data')
                    state = 1
                # except queue.Empty:
                except asyncio.TimeoutError:
                    speech_chunk = np.zeros(self.chunk_size, dtype=np.float32)
                    state = 0
                yield Data(
                    data=speech_chunk,
                    state=state,
                )
        return producer

    # pipeline2
    def speech_batch_consumer(self) -> AsyncConsumer:
        async def consume_fn(data: Data, processed_data: Data = None):
            await self.speech_frames_2.put(data)
            speech_chunk = data.data
            self.speech_frame_batch.append(speech_chunk)
            if len(self.speech_frame_batch) == self.batch_size*2+20:
                # speech_feature, self.speech_frame_batch = self.model.encode_speech_feature(self.speech_frame_batch)
                speech_feature, self.speech_frame_batch = encode(self.speech_frame_batch, self.fps)
                await self.speech_features.put(
                    Data(data=speech_feature)
                )
            return data

        handler = None
        return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)

    def speech_feature_producer(self):
        async def producer():
            while not self.stop_event.is_set() or not self.speech_features.empty():
                try:
                    speech_feature_data = await asyncio.wait_for(self.speech_features.get(), timeout=self.timeout_feature)
                    yield speech_feature_data
                # except queue.Empty:
                except asyncio.TimeoutError:
                    continue
        return producer

    # pipeline3
    def speech_feature_consumer(self) -> AsyncConsumer:
        async def consume_fn(data: Data, processed_data: Data = None):
            speech_feature_batch = data.get('data')
            speech_frames = []
            silence = True
            for _ in range(self.batch_size * 2):
                speech_frame_data = await asyncio.wait_for(self.speech_frames_2.get(), timeout=self.timeout_data)
                speech_frame, state = speech_frame_data.get('data'), speech_frame_data.get('state')
                speech_frames.append(speech_frame_data)
                if state == 1:
                    silence = False
            if silence:
                for i in range(self.batch_size):
                    avatar_frame, speech_frame, frame_idx = None, speech_frames[i*2: i*2+2], self.mirror_index(self.frame_index)
                    await self.put_res_frame(avatar_frame, speech_frame, frame_idx)
                    self.update_index(1)
            else:
                face_img_batch = []
                for i in range(self.batch_size):
                    idx = self.mirror_index(self.frame_index+i)
                    face_img = self.face_list_cycle[idx]
                    face_img_batch.append(face_img)
                face_img_batch, speech_feature_batch = np.asarray(face_img_batch), np.asarray(speech_feature_batch)
                face_img_masked = face_img_batch.copy()
                face_img_masked[:, face_img_batch[0].shape[0]//2:] = 0
                face_img_batch = np.concatenate((face_img_masked, face_img_batch), axis=3)/255.
                speech_feature_batch = np.reshape(speech_feature_batch,
                                                  [speech_feature_batch.shape[0], speech_feature_batch.shape[1],
                                                   speech_feature_batch.shape[2], 1])

                face_img_batch = torch.FloatTensor(np.transpose(face_img_batch, (0, 3, 1, 2))).to(DEVICE)
                speech_feature_batch = torch.FloatTensor(np.transpose(speech_feature_batch, (0, 3, 1, 2))).to(DEVICE)

                with torch.no_grad():
                    pred_img_batch = self.model(speech_feature_batch, face_img_batch)
                pred_img_batch = pred_img_batch.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                for i, avatar_frame in enumerate(pred_img_batch):
                    speech_frame, frame_idx = speech_frames[i*2: i*2+2], self.mirror_index(self.frame_index)
                    await self.put_res_frame(avatar_frame, speech_frame, frame_idx)
                    self.update_index(1)

            return data

        handler = None
        return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)

    def result_frame_producer(self):
        async def producer():
            while not self.stop_event.is_set() or not self.res_frames.empty():
                try:
                    res_frame = await asyncio.wait_for(self.res_frames.get(), timeout=self.timeout_frame)
                    yield Data(
                        data=res_frame,
                    )
                # except queue.Empty:
                except asyncio.TimeoutError:
                    continue
        return producer

   # pipeline4
    def result_frame_consumer(self) -> AsyncConsumer:
        async def consume_fn(data: Data, processed_data: Data = None):
            avatar_frame, speech_frame, frame_idx = data.get('data')
            if speech_frame[0].get('state') == 0 and speech_frame[1].get('state') == 0:  # 全为静音数据，只需要取fullimg
                self.speaking = False
                new_frame = self.frame_list_cycle[frame_idx]
            else:
                new_frame = self.make_avatar_frame(frame_idx, avatar_frame)

            image = new_frame
            image[0, :] &= 0xFE
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            print(new_frame)

            media_player.add_video_frame(image)

            for frame_data in speech_frame:
                frame, state = frame_data.get('data'), frame_data.get('state')
                frame = (frame * 32767).astype(np.int16)

                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = self.sample_rate
                print(new_frame)

                media_player.add_audio_frame(frame)

            return data

        handler = None
        return AsyncConsumerFactory.with_consume_fn(consume_fn, handler=handler)



if __name__ == '__main__':
    f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr.wav'
    c_f = '../../../checkpoints/wav2lip.pth'
    stop_event = asyncio.Event()
    model = load_model(c_f)
    player = Player(
        model=model,
        avatar=load_avatar(f),
        fps=50,
        sample_rate=SAMPLE_RATE,
        batch_size=16,
        stop_event=stop_event,
        timeout_data=0.01,
        timeout_feature=1,
        timeout_frame=1,
    )

    pipeline1 = AsyncPipeline(
        producer=soundfile_producer(s_f, chunk_size_or_fps=player.fps),
        consumers=[player.speech_data_consumer()]
    )
    pipeline2 = AsyncPipeline(
        producer=player.speech_data_producer(),
        consumers=[player.speech_batch_consumer()]
    )
    pipeline3 = AsyncPipeline(
        producer=player.speech_feature_producer(),
        consumers=[player.speech_feature_consumer()]
    )
    pipeline4 = AsyncPipeline(
        producer=player.result_frame_producer(),
        consumers=[player.result_frame_consumer()]
    )

    pipelines = [pipeline1, pipeline2, pipeline3, pipeline4]
    async def run_pipelines():
        tasks = []
        for p in pipelines:
            tasks.append(p.start())
        for p in pipelines:
            tasks.append(p.stop())
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run_pipelines())
