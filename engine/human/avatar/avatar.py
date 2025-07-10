import copy
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from engine.config import PlayerConfig, AvatarProcessorConfig
from engine.utils import EasyDict
from engine.utils.data import Data


class Avatar(EasyDict):
    def __init__(self, avatar_resource: dict):
        super().__init__(avatar_resource)

        assert self.frame_cycle is not None
        assert self.bbox_cycle is not None

        self.frame_index = 0
        self.frame_count = len(self.frame_cycle)

    def reset(self):
        self.frame_index = 0

    def mirror_frame_index(self, index):
        turn = index // self.frame_count
        res = index % self.frame_count
        if turn % 2 == 0:
            return res
        else:
            return self.frame_count - res - 1

    def update_frame_index(self, n):
        self.frame_index += n

    def set_frame_index(self, frame_index):
        self.frame_index = frame_index

    def get_frame(self, frame_index) -> np.ndarray:
        return self.frame_cycle[frame_index]

    def get_bbox(self, frame_index) -> np.ndarray:
        return self.bbox_cycle[frame_index]

    def get_any_data(self, frame_index, data_type: str) -> np.ndarray:
        return getattr(self, data_type)[frame_index]

    def get_next_frame(self):
        frame_index = self.mirror_frame_index(self.frame_index)
        frame = self.get_frame(frame_index)
        self.update_frame_index(1)
        return frame

    def render_frame(self, pred) -> np.ndarray:
        frame_index = self.mirror_frame_index(self.frame_index)
        frame = copy.deepcopy(self.get_frame(frame_index))
        bbox = self.get_bbox(frame_index)
        y1, y2, x1, x2 = bbox
        pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))
        frame[y1:y2, x1:x2] = pred
        self.update_frame_index(1)
        return frame


class AvatarProcessor:
    """
    todo 对于每一帧, 做各种后处理
    """
    def __init__(self, config: AvatarProcessorConfig):
        self.config = config

    def process(
            self,
            frame: np.ndarray,
            *args,
            **kwargs,
    ) -> np.ndarray:
        return frame


class AvatarModelWrapper(nn.Module):
    def load_backbone(self) -> None:
        """
        加载 backbone 模型
        :return: backbone
        """
        pass

    def inference(
        self,
        audio_feature_batch: List[np.ndarray],
        config: PlayerConfig,
        **kwargs,
    ) -> np.ndarray:
        """
        通过音频特征和人脸图像, 预测口型图像
        :param audio_feature_batch:
        :param face_img_batch:
        :param config:
        :return:
        """
        pass

