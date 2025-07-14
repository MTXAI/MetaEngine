from typing import List

import numpy as np
from transformers import WhisperModel, WhisperPreTrainedModel

from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG
from engine.human.avatar import AvatarModelWrapper, AvatarResource
from models.musetalk.models.unet import UNet, PositionalEncoding
from models.musetalk.models.vae import VAE
from models.musetalk.utils.audio_processor import AudioProcessor


class MuseTalkWrapper(AvatarModelWrapper):
    def __init__(
            self,
            unet_dir,
            vae_dir,
            whisper_dir,
    ):
        super().__init__()
        self.unet_dir = unet_dir
        self.vae_dir = vae_dir
        self.whisper_dir = whisper_dir
        self.unet = None
        self.vae = None
        self.pe = None
        self.whisper = None
        self.audio_processor = None
        self.load_backbone()

    def load_backbone(self):
        self.vae = VAE(
            model_path=self.unet_dir,
            use_float16=DEFAULT_RUNTIME_CONFIG.use_float16,
            device=DEFAULT_RUNTIME_CONFIG.device,
        )
        self.unet = UNet(
            unet_config="",
            model_path=self.unet_dir,
            use_float16=DEFAULT_RUNTIME_CONFIG.use_float16,
            device=DEFAULT_RUNTIME_CONFIG.device,
        )
        self.pe = PositionalEncoding(d_model=384)
        weight_dtype = self.unet.model.dtype
        whisper: WhisperPreTrainedModel = WhisperModel.from_pretrained(self.whisper_dir)
        if DEFAULT_RUNTIME_CONFIG.use_float16:
            whisper = whisper.half()
        whisper = whisper.to(device=DEFAULT_RUNTIME_CONFIG.device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)
        self.whisper = whisper
        self.audio_processor = AudioProcessor(feature_extractor_path=self.whisper_dir)

    def inference(
        self,
        audio_chunk_batch: List[np.ndarray],
        avatar_resource: AvatarResource,
        config: PlayerConfig,  # todo 修改 config 为指定的
        **kwargs,
    ) -> np.ndarray:
        """
        通过音频特征和人脸图像, 预测口型图像
        :param audio_chunk_batch:
        :param avatar_resource:
        :param config:
        :return:
        """
        pass