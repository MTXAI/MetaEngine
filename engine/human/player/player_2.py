import asyncio
import copy
import queue
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from av import AudioFrame, VideoFrame
from torch import nn

from engine.config import DEFAULT_RUNTIME_CONFIG, PlayerConfig
from engine.human.avatar.wav2lip import load_avatar, load_model
from engine.human.utils.data import Data
from engine.human.voice.asr import soundfile_producer
from engine.utils.pipeline import AsyncPipeline
from models.wav2lip.audio import melspectrogram

MEL_STEP_SIZE = 16
MEL_IDX_MULTIPLIER = 80. * 2
SAMPLE_RATE = 16000

mp.set_start_method(DEFAULT_RUNTIME_CONFIG.start_method, force=True)


def encode(speech_frame_batch, fps=50):
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
            config: PlayerConfig,
            model: nn.Module,
            # model.load_state_dict, model.warmup, model.inference
            avatar: Tuple,
            stop_event: threading.Event,
    ):
        self.config = config
        self.fps = config.fps
        self.sample_rate = config.sample_rate
        self.batch_size = config.batch_size
        self.chunk_size = int(self.sample_rate / self.fps)
        self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle = avatar
        self.model = model
        self.stop_event = stop_event


        self.speech_frames = mp.Queue()
        self.speech_frames_2 = mp.Queue()
        self.speech_features = mp.Queue(2)
        self.res_frames = mp.Queue(self.batch_size*2)  # avatar_frame, speech_frame, frame_idx

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

    def put_res_frame(self, avatar_frame, speech_frame, frame_idx):
        self.res_frames.put((avatar_frame, speech_frame, frame_idx))

    def put_speech_data(self, data: Data):
        self.speech_frames.put(data)

    def get_speech_data(self):
        try:
            speech_chunk_data = self.speech_frames.get(timeout=self.timeout_data)
            speech_chunk = speech_chunk_data.get('data')
            state = 1
        except queue.Empty:
            speech_chunk = np.zeros(self.chunk_size, dtype=np.float32)
            state = 0
        return Data(
                    data=speech_chunk,
                    state=state,
                )

    def warm_up(self):
        for _ in range(20):
            data = self.get_speech_data()
            self.speech_frames_2.put(data)
            speech_chunk = data.data
            self.speech_frame_batch.append(speech_chunk)
        for _ in range(10):
            self.speech_frames_2.get()

    def make_speech_feature(self):
        for _ in range(self.batch_size*2):
            data = self.get_speech_data()
            self.speech_frames_2.put(data)
            speech_chunk = data.data
            self.speech_frame_batch.append(speech_chunk)
        speech_feature, self.speech_frame_batch = encode(self.speech_frame_batch, self.fps)
        self.speech_features.put(
            Data(
                data=speech_feature,
            )
        )

    def model_inference(self):
        while not self.stop_event.is_set():
            try:
                speech_feature_data = self.speech_features.get(timeout=self.timeout_feature)
            except queue.Empty:
                continue
            speech_feature_batch = speech_feature_data.get('data')
            speech_frames = []
            silence = True
            for _ in range(self.batch_size * 2):
                speech_frame_data = self.speech_frames_2.get(timeout=self.timeout_data)
                speech_frame, state = speech_frame_data.get('data'), speech_frame_data.get('state')
                speech_frames.append(speech_frame_data)
                if state == 1:
                    silence = False
            if silence:
                for i in range(self.batch_size):
                    avatar_frame, speech_frame, frame_idx = None, speech_frames[i*2: i*2+2], self.mirror_index(self.frame_index)
                    self.put_res_frame(avatar_frame, speech_frame, frame_idx)
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
                st = time.time()
                with torch.no_grad():
                    pred_img_batch = self.model(speech_feature_batch, face_img_batch)
                et = time.time()
                print(f"Time: {et - st}", pred_img_batch.shape)
                pred_img_batch = pred_img_batch.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                for i, avatar_frame in enumerate(pred_img_batch):
                    speech_frame, frame_idx = speech_frames[i * 2: i * 2 + 2], self.mirror_index(self.frame_index)
                    self.put_res_frame(avatar_frame, speech_frame, frame_idx)
                    self.update_index(1)

    def frame_process(self):
        while not self.stop_event.is_set():
            try:
                res_frame = self.res_frames.get(timeout=self.timeout_frame)
            except queue.Empty:
                continue
            avatar_frame, speech_frame, frame_idx = res_frame
            if speech_frame[0].get('state') == 0 and speech_frame[1].get('state') == 0:  # 全为静音数据，只需要取fullimg
                new_frame = self.frame_list_cycle[frame_idx]
            else:
                new_frame = self.make_avatar_frame(frame_idx, avatar_frame)

            image = new_frame
            image[0, :] &= 0xFE
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            print(new_frame)

            for frame_data in speech_frame:
                frame, state = frame_data.get('data'), frame_data.get('state')
                frame = (frame * 32767).astype(np.int16)

                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = self.sample_rate
                print(new_frame)

    def run(self):
        threading.Thread(target=self.frame_process).start()
        threading.Thread(target=self.model_inference).start()
        while not self.stop_event.is_set():
            self.make_speech_feature()


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
    player.warm_up()

    threading.Thread(target=player.run).start()

    def consume_fn(data):
        player.put_speech_data(data)
        return data

    pipeline = AsyncPipeline(
        producer=soundfile_producer(s_f, chunk_size_or_fps=player.fps),
        consumers=[
            AsyncConsumerFactory.with_consume_fn(consume_fn),
        ]
    )
    runner = AsyncPipelineRunner()
    runner.add_pipeline(pipeline)
    runner.run()
