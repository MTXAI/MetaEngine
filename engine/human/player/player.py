import asyncio
import copy
import queue
import threading
import traceback
from concurrent.futures import Future
from queue import Queue
from typing import Tuple
import time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from av import AudioFrame, VideoFrame

from engine.config import DEFAULT_RUNTIME_CONFIG, PlayerConfig
from engine.human.avatar.avatar import ModelWrapper
from engine.human.utils.data import Data
from engine.utils.pool import ThreadPool, TaskInfo

mp.set_start_method(DEFAULT_RUNTIME_CONFIG.start_method, force=True)


class Player:
    def __init__(
            self,
            config: PlayerConfig,
            model: ModelWrapper,
            avatar: Tuple,
            thread_pool: ThreadPool,
            # video_track: ,
            # audio_track: ,
    ):
        print(config)
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.batch_size = config.batch_size
        self.chunk_size = int(self.sample_rate / self.fps)
        self.frame_interval = config.frame_interval
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        self.model = model
        self.stop_event = threading.Event()
        self.event_loop = asyncio.get_event_loop()

        # 设置队列最大大小，防止无限阻塞
        self.input_queue = Queue()
        self.frame_queue = Queue()
        self.feature_queue = Queue()
        self.output_queue = Queue(maxsize=self.batch_size * 2)

        self.frame_batch = []
        self.frame_count = len(self.frame_list_cycle)
        self.frame_index = 0
        self.thread_pool = thread_pool
        self.monitor_thread = None
        self.monitor_running = False
        self.debug = False

    def mirror_index(self, index):
        turn = index // self.frame_count
        res = index % self.frame_count
        if turn % 2 == 0:
            return res
        else:
            return self.frame_count - res - 1

    def update_index(self, n):
        self.frame_index += n

    def warmup(self):
        for _ in range(self.config.warmup_iters):
            frame = self.get_audio_frame()
            self.frame_queue.put(frame)
            self.frame_batch.append(frame.get("data"))
        self._print_queue_status("Warmup completed")

    def put_audio_data(self, data: Data):
        self.input_queue.put(data, timeout=1)

    def get_audio_frame(self) -> Data:
        try:
            data = self.input_queue.get(timeout=0.01)
            audio_frame = data.get('data')
            state = 1
        except queue.Empty:
            audio_frame = np.zeros(self.chunk_size, dtype=np.float32)
            state = 0
        return Data(
            data=audio_frame,
            state=state,
        )

    def process_audio_frame_worker(self):
        while not self.stop_event.is_set():
            try:
                for _ in range(self.batch_size * 2):
                    frame = self.get_audio_frame()
                    self.frame_queue.put(frame, timeout=0.1)  # 添加超时
                    self.frame_batch.append(frame.get("data"))
                    time.sleep(self.frame_interval)
                audio_feature_batch = self.model.encode_audio_feature(self.frame_batch, self.config)
                self.frame_batch = self.frame_batch[self.batch_size * 2:]
                self.feature_queue.put(
                    Data(data=audio_feature_batch),
                    timeout=0.1
                )

                # # 每处理一批数据后打印队列状态
                # self._print_queue_status("After processing audio frame batch")
            except Exception as e:
                print(f"Audio processing error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def infer_video_frame_worker(self):
        count = 0
        counttime = 0
        while not self.stop_event.is_set():
            try:
                try:
                    audio_feature_data = self.feature_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                audio_feature_batch = audio_feature_data.get("data")
                audio_frames = []
                for _ in range(self.batch_size * 2):
                    frame_data = self.frame_queue.get(timeout=0.1)
                    audio_frames.append(frame_data)

                silence = all(frame.get("state") == 0 for frame in audio_frames)
                t = time.perf_counter()


                if silence:
                    for i in range(self.batch_size):
                        video_frame = None
                        audio_frame = audio_frames[i * 2:i * 2 + 2]
                        frame_index = self.mirror_index(self.frame_index)
                        self.output_queue.put(
                            (video_frame, audio_frame, frame_index),
                            timeout=0.1
                        )
                        self.update_index(1)
                else:
                    face_img_batch = []
                    for i in range(self.batch_size):
                        frame_index = self.mirror_index(self.frame_index + i)
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
                        continue
                    # 处理推理结果
                    for i, video_frame in enumerate(pred_img_batch):
                        audio_frame = audio_frames[i * 2:i * 2 + 2]
                        frame_index = self.mirror_index(self.frame_index)
                        self.output_queue.put(
                            (video_frame, audio_frame, frame_index),
                            timeout=0.1
                        )
                        self.update_index(1)
                if self.debug:
                    count += self.batch_size
                    counttime += (time.perf_counter() - t)
                    # _totalframe += 1
                    if count >= 100:
                        print(f"------actual avg infer fps:{count / counttime:.4f}")
                        count = 0
                        counttime = 0

                        # 每批推理完成后打印队列状态
                        self._print_queue_status("After inference batch")
            except Exception as e:
                print(f"Video inference error: {e}")
                traceback.print_exc()
                time.sleep(0.1)  # 出错后短暂休眠

    def process_output_frames_worker(self):
        while not self.stop_event.is_set():
            try:
                # 非阻塞获取输出帧
                try:
                    output_frame = self.output_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                video_frame, audio_frame, frame_index = output_frame

                # 处理视频帧
                if audio_frame[0].get('state') == 0 and audio_frame[1].get('state') == 0:
                    new_frame = self.frame_list_cycle[frame_index]
                else:
                    new_frame = copy.deepcopy(self.frame_list_cycle[frame_index])
                    bbox = self.coord_list_cycle[frame_index]
                    y1, y2, x1, x2 = bbox

                    if video_frame is not None:
                        video_frame = cv2.resize(video_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                        new_frame[y1:y2, x1:x2] = video_frame

                # 创建视频帧
                image = new_frame
                image[0, :] &= 0xFE  # 确保第一行是偶数，避免某些视频问题
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                # asyncio.run_coroutine_threadsafe(self.video_track.put_frame(new_frame), self.event_loop)

                # 处理音频帧
                for frame_data in audio_frame:
                    frame, state = frame_data.get('data'), frame_data.get('state')
                    frame = (frame * 32767).astype(np.int16)
                    new_audio_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_audio_frame.planes[0].update(frame.tobytes())
                    new_audio_frame.sample_rate = self.sample_rate
                    # print(new_audio_frame)
                    # asyncio.run_coroutine_threadsafe(self.audio_track.put_frame(new_frame), self.event_loop)

                # # 每处理一帧后打印队列状态
                # self._print_queue_status("After processing output frame")
            except Exception as e:
                print(f"Output processing error: {e}")
                traceback.print_exc()
                time.sleep(0.1)  # 出错后短暂休眠

    def start(self):
        self.warmup()

        # 启动队列监控线程
        if self.debug:
            self.monitor_running = True
            self.monitor_thread = threading.Thread(
                target=self._queue_monitor,
                daemon=True
            )
            self.monitor_thread.start()

        self.thread_pool.submit(
            self.process_audio_frame_worker,
            task_info=TaskInfo(name="player.process_audio_frame_worker")
        ),
        self.thread_pool.submit(
            self.infer_video_frame_worker,
            task_info=TaskInfo(name="player.infer_video_frame_worker")
        ),
        self.thread_pool.submit(
            self.process_output_frames_worker,
            task_info=TaskInfo(name="player.process_output_frames_worker")
        )
        self._print_queue_status("Player started")

    def shutdown(self):
        # 设置停止事件
        self.stop_event.set()
        self.monitor_running = False  # 停止监控线程

        # 清空队列，避免阻塞
        queues = [self.input_queue, self.frame_queue, self.feature_queue, self.output_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    continue
        self._print_queue_status("Player stopped, queues cleared")

    def _print_queue_status(self, message: str = "Queue status"):
        if not self.debug:
            return
        """打印当前所有队列的大小"""
        input_size = self.input_queue.qsize()
        frame_size = self.frame_queue.qsize()
        feature_size = self.feature_queue.qsize()
        output_size = self.output_queue.qsize()

        print(f"[{time.strftime('%H:%M:%S')}] {message}:")
        print(f"  Input queue size: {input_size}")
        print(f"  Frame queue size: {frame_size}")
        print(f"  Feature queue size: {feature_size}")
        print(f"  Output queue size: {output_size}")
        print()

    def _queue_monitor(self):
        """定期监控队列大小的后台线程"""
        while self.monitor_running:
            self._print_queue_status("Periodic queue monitor")
            time.sleep(5)  # 每5秒监控一次


if __name__ == '__main__':
    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.runtime import thread_pool
    import time

    f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr.wav'
    c_f = '../../../checkpoints/wav2lip.pth'
    model = Wav2LipWrapper(c_f)

    # 创建Player实例并启动
    player = Player(WAV2LIP_PLAYER_CONFIG, model, load_avatar(f), thread_pool)
    player.start()

    # 设置音频数据生产者
    def consume_fn(data):
        player.put_audio_data(data)
        return data

    from engine.human.voice.asr import soundfile_producer
    from engine.utils.pipeline import Pipeline

    pipeline = Pipeline(
        producer=soundfile_producer(s_f, chunk_size_or_fps=player.fps),
        consumer=consume_fn,
    )

    producer_task = thread_pool.submit(
        pipeline.produce_worker,
        task_info=TaskInfo(name="Task producer")
    )
    consumer_task = thread_pool.submit(
        pipeline.consume_worker,
        task_info=TaskInfo(name="Task consumer")
    )

    while True:
        if not producer_task.done() or not consumer_task.done():
            time.sleep(1)
            continue
        else:
            break
    # pipeline.shutdown()
    # player.shutdown()
    thread_pool.shutdown()

