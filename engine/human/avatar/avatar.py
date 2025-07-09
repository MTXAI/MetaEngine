from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from engine.config import PlayerConfig
from engine.utils.data import Data


class AvatarModelWrapper(nn.Module):
    def gen_avatar(self):
        """
        生成 avatar 资源
        :return:
        """
        pass

    def load_backbone(self) -> nn.Module:
        """
        加载 backbone 模型
        :return: backbone
        """
        pass

    def load_avatar(self) -> Tuple[List, List, List]:
        """
        加载 avatar 资源
        :return: frame_list_cycle face_list_cycle coord_list_cycle
        """
        pass

    def encode_audio_feature(self, audio_data_batch: List[np.ndarray], config: PlayerConfig, **kwargs) -> List[np.ndarray]:
        """
        编码音频特征
        :param audio_data_batch: 音频数据, 其中 audio_data_batch 的数据长度不一定等于 batch size
        :param config:
        :return:
        """
        pass

    def inference(self, audio_feature_batch: torch.Tensor, face_img_batch: torch.Tensor, config: PlayerConfig):
        """
        通过音频特征和人脸图像, 预测口型图像
        :param audio_feature_batch:
        :param face_img_batch:
        :param config:
        :return:
        """
        pass

