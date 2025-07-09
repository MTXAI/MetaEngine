from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from engine.config import PlayerConfig
from engine.human.avatar import AvatarModelWrapper


class MuseTalkWrapper(AvatarModelWrapper):
    def __init__(self, ckpt_path, avatar_path):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.avatar_path = avatar_path
        self.backbone = None

    def gen_avatar(self):
        pass

    def load_backbone(self) -> nn.Module:
        pass

    def load_avatar(self) -> Tuple[List, List, List]:
        pass

    def encode_audio_feature(self, audio_data_batch: List[np.ndarray], config: PlayerConfig, **kwargs) -> List[np.ndarray]:
        pass

    def inference(self, audio_feature_batch: torch.Tensor, face_img_batch: torch.Tensor, config: PlayerConfig):
        pass
