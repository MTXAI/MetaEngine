from typing import List

import torch
from torch import nn

from engine.config import PlayerConfig
from engine.human.utils.data import Data


class AvatarModelWrapper(nn.Module):
    def encode_audio_feature(self, frame_batch: List[Data], config: PlayerConfig):
        pass

    def inference(self, audio_feature_batch: torch.Tensor, face_img_batch: torch.Tensor, config: PlayerConfig):
        pass

