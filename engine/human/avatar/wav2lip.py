import glob
import logging
import os
import pickle
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG
from engine.human.avatar.avatar import AvatarModelWrapper
from models.wav2lip.audio import melspectrogram
from models.wav2lip.models import Wav2Lip


def load_model(path):
    model = Wav2Lip()
    logging.info("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(DEFAULT_RUNTIME_CONFIG.device)
    return model.eval()

def _read_imgs(img_list):
    frames = []
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def load_avatar(avatar_path):
    full_imgs_path = f"{avatar_path}/full_imgs"
    face_imgs_path = f"{avatar_path}/face_imgs"
    coords_path = f"{avatar_path}/coords.pkl"

    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = _read_imgs(input_img_list)
    # self.imagecache = ImgCache(len(self.coord_list_cycle),self.full_imgs_path,1000)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = _read_imgs(input_face_list)

    return frame_list_cycle ,face_list_cycle ,coord_list_cycle

class Wav2LipWrapper(AvatarModelWrapper):
    def __init__(self, path):
        super().__init__()
        self.model = load_model(path)

    def encode_audio_feature(self, frame_batch: List[np.array], config: PlayerConfig):
        frames = np.concatenate(frame_batch)
        mel = melspectrogram(frames)

        batch_size = config.batch_size
        mel_step_size = batch_size
        i = 0
        audio_feature_batch = []
        while i < batch_size:
            start_idx = 0
            if start_idx + mel_step_size > len(mel[0]):
                audio_feature_batch.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                audio_feature_batch.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1
        return audio_feature_batch

    def inference(self, audio_feature_batch: torch.Tensor, face_img_batch: torch.Tensor, config: PlayerConfig):
        face_img_masked = face_img_batch.clone()
        face_img_masked[:, face_img_batch[0].shape[0] // 2:] = 0
        face_img_batch = torch.cat((face_img_masked, face_img_batch), dim=3) / 255.
        if len(audio_feature_batch.shape) < 4:
            audio_feature_batch = audio_feature_batch.unsqueeze(-1)
        audio_feature_batch = audio_feature_batch.permute(0, 3, 1, 2)
        face_img_batch = face_img_batch.permute(0, 3, 1, 2)
        pred_img_batch = self.model(audio_feature_batch, face_img_batch)
        pred_img_batch = pred_img_batch.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred_img_batch

