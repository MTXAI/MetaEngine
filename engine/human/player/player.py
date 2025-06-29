import copy
import queue
import threading
from queue import Queue
from typing import Tuple

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
            thread_pool: ThreadPool
    ):
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.batch_size = config.batch_size
        self.chunk_size = int(self.sample_rate / self.fps)
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        self.model = model
        self.stop_event = threading.Event()

        self.input_queue = Queue()
        self.frame_queue = Queue()
        self.feature_queue = Queue(2)
        self.output_queue = Queue(self.batch_size*2)

        self.frame_batch = []
        self.frame_count = len(self.frame_list_cycle)
        self.frame_index = 0
        self.thread_pool = thread_pool

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
            self.frame_batch.append(frame)
        for _ in range(self.config.warmup_iters // 2):
            self.output_queue.get()

    def put_audio_data(self, data: Data):
        self.input_queue.put(data)

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
            for _ in range(self.batch_size * 2):
                frame = self.get_audio_frame()
                self.frame_queue.put(frame)
                self.frame_batch.append(frame.data)
            audio_feature_batch = self.model.encode_audio_feature(self.frame_batch, self.config)
            self.frame_batch = self.frame_batch[self.batch_size * 2:]
            self.feature_queue.put(
                Data(
                    data=audio_feature_batch,
                )
            )

    def infer_video_frame_worker(self):
        while not self.stop_event.is_set():
            try:
                audio_feature_data = self.feature_queue.get(timeout=1)
                audio_feature_batch = audio_feature_data.get("data")
            except queue.Empty:
                continue

            audio_frames = []
            silence = True
            for _ in range(self.batch_size * 2):
                frame_data = self.frame_queue.get(timeout=0.1)
                state = frame_data.get("state")
                audio_frames.append(frame_data)
                if state == 1:
                    silence = False
            if silence:
                for i in range(self.batch_size):
                    video_frame, audio_frame, frame_index = None, audio_frames[i*2:i*2+2], self.mirror_index(self.frame_index)
                    self.output_queue.put(
                        (video_frame, audio_frame, frame_index)
                    )
                    self.update_index(1)
            else:
                face_img_batch = []
                for i in range(self.batch_size):
                    frame_index = self.mirror_index(self.frame_index+i)
                    face_img = self.frame_list_cycle[frame_index]
                    face_img_batch.append(face_img)
                face_img_batch, audio_feature_batch = np.asarray(face_img_batch), np.asarray(audio_feature_batch)
                face_img_batch = torch.FloatTensor(face_img_batch).to(DEFAULT_RUNTIME_CONFIG.device)
                audio_feature_batch = torch.FloatTensor(audio_feature_batch).to(DEFAULT_RUNTIME_CONFIG.device)

                with torch.no_grad():
                    pred_img_batch = self.model.inference(audio_feature_batch, face_img_batch, self.config)

                for i, video_frame in enumerate(pred_img_batch):
                    audio_frame, frame_index = audio_frames[i*2:i*2+2], self.mirror_index(self.frame_index)
                    self.output_queue.put(
                        (video_frame, audio_frame, frame_index)
                    )
                    self.update_index(1)

    def process_output_frames_worker(self):
        while not self.stop_event.is_set():
            try:
                output_frame = self.output_queue.get(timeout=1)
            except queue.Empty:
                continue
            video_frame, audio_frame, frame_index = output_frame

            if audio_frame[0].get('state') == 0 and audio_frame[1].get('state') == 0:
                new_frame = self.frame_list_cycle[frame_index]
            else:
                new_frame = copy.deepcopy(self.frame_list_cycle[frame_index])
                bbox = self.coord_list_cycle[frame_index]
                y1, y2, x1, x2 = bbox
                video_frame = cv2.resize(video_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                new_frame[y1:y2, x1:x2] = video_frame

            image = new_frame
            image[0, :] &= 0xFE
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            print(new_frame)

            for frame_data in audio_frame:
                frame, state = frame_data.get('data'), frame_data.get('state')
                frame = (frame * 32767).astype(np.int16)

                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = self.sample_rate
                print(new_frame)

    def start(self):
        self.warmup()
        self.thread_pool.submit(
            self.process_audio_frame_worker,
            task_info=TaskInfo(
                name="player.process_audio_frame_worker",
            )
        )
        self.thread_pool.submit(
            self.infer_video_frame_worker,
            task_info=TaskInfo(
                name="player.infer_video_frame_worker",
            )
        )
        self.thread_pool.submit(
            self.process_output_frames_worker,
            task_info=TaskInfo(
                name="player.process_output_frames_worker",
            )
        )

    def stop(self):
        self.stop_event.set()


if __name__ == '__main__':
    from engine.config import WAV2LIP_PLAYER_CONFIG
    from engine.human.avatar.wav2lip import Wav2LipWrapper, load_avatar
    from engine.utils.pool import ThreadPool
    import threading
    from engine.human.voice.asr import soundfile_producer
    import trio
    from engine.utils.pipeline import AsyncPipeline, Pipeline

    f = '../../../avatars/wav2lip256_avatar1'
    s_f = '../../../tests/test_datas/asr.wav'
    c_f = '../../../checkpoints/wav2lip.pth'
    model = Wav2LipWrapper(
        c_f
    )
    pool = ThreadPool(
        max_workers=4,
        max_queue_size=10,
    )
    player = Player(
        WAV2LIP_PLAYER_CONFIG,
        model,
        load_avatar(f),
        pool
    )
    player.start()

    def consume_fn(data):
        player.put_audio_data(data)
        return data
    pipeline = Pipeline(
        producer=soundfile_producer(s_f, chunk_size_or_fps=player.fps),
        consumer=consume_fn,
    )
    pool.submit(
        pipeline.produce_worker,
        task_info=TaskInfo(
            name="Task producer",
        ),
    )
    pool.submit(
        pipeline.consume_worker,
        task_info=TaskInfo(
            name="Task consumer",
        ),
    )

    pool.shutdown()
