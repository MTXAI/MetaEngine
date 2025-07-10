import glob
import os
import pickle
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG, frame_multiple
from engine.human.avatar.avatar import AvatarModelWrapper, Avatar
from models.wav2lip import Wav2Lip
from models.wav2lip.audio import melspectrogram
from models.wav2lip.hparams import hparams


def _read_imgs(img_list):
    frames = []
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def gen_avatar() -> Avatar:
    pass


# todo 实现 register, 注册 load, gen 和 模型
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

    return Avatar(
        dict(
            frame_cycle=frame_list_cycle,
            bbox_cycle=coord_list_cycle,
            face_cycle=face_list_cycle,
        )
    )


class Wav2LipWrapper(AvatarModelWrapper):
    def __init__(self, ckpt_path: str, avatar: Avatar):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.avatar: Avatar = avatar
        self.backbone = None
        self.load_backbone()

    def load_backbone(self):
        model = Wav2Lip()
        if DEFAULT_RUNTIME_CONFIG.use_fp16:
            model = model.half()
        model = model.to(DEFAULT_RUNTIME_CONFIG.device)
        checkpoint = torch.load(self.ckpt_path, map_location=lambda storage, loc: storage)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        self.backbone = model.eval()

    def _encode_audio_feature(self, frame_batch: List[np.array], config: PlayerConfig, **kwargs) -> List[np.ndarray]:
        # expect 5120
        frames = np.concatenate(frame_batch)
        mel = melspectrogram(frames)

        batch_size = config.batch_size
        frame_multiple = config.frame_multiple
        mel_step_size = batch_size
        i = 0
        audio_feature_batch = []
        while i < batch_size:
            start_idx = i*frame_multiple
            if start_idx + mel_step_size > len(mel[0]):
                audio_feature_batch.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                audio_feature_batch.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1
        return audio_feature_batch

    def inference(
        self,
        audio_chunk_batch: List[np.ndarray],
        config: PlayerConfig,
        **kwargs
    ) -> np.ndarray:
        audio_feature_batch = self._encode_audio_feature(audio_chunk_batch, config)
        audio_feature_batch = np.asarray(audio_feature_batch)
        audio_feature_batch = torch.FloatTensor(audio_feature_batch).to(DEFAULT_RUNTIME_CONFIG.device)

        face_img_batch = []
        for i in range(config.batch_size):
            frame_index = self.avatar.mirror_frame_index(self.avatar.frame_index + i)
            face_img = self.avatar.get_any_data(frame_index, data_type="face_cycle")
            face_img_batch.append(face_img)
        face_img_batch = np.asarray(face_img_batch)
        face_img_batch = torch.FloatTensor(face_img_batch).to(DEFAULT_RUNTIME_CONFIG.device)

        # expect torch.Size([16, 80, 16]) torch.Size([16, 256, 256, 3])
        face_img_masked = face_img_batch.clone()
        face_img_masked[:, face_img_batch[0].shape[0] // 2:] = 0
        face_img_batch = torch.cat((face_img_masked, face_img_batch), dim=3) / 255.
        if len(audio_feature_batch.shape) < 4:
            audio_feature_batch = audio_feature_batch.unsqueeze(-1)
        audio_feature_batch = audio_feature_batch.permute(0, 3, 1, 2)
        face_img_batch = face_img_batch.permute(0, 3, 1, 2)
        pred_img_batch = self.backbone(audio_feature_batch, face_img_batch)
        pred_img_batch = pred_img_batch.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred_img_batch

