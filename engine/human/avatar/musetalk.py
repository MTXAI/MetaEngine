from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from transformers import WhisperModel, WhisperPreTrainedModel

from engine.config import PlayerConfig, DEFAULT_RUNTIME_CONFIG
from engine.human.avatar import AvatarModelWrapper
from engine.human.avatar.avatar import Avatar
from models.musetalk.models.unet import UNet, PositionalEncoding
from models.musetalk.models.vae import VAE
from models.musetalk.utils.audio_processor import AudioProcessor


class MuseTalkWrapper(AvatarModelWrapper):
    def __init__(
            self,
            unet_dir,
            vae_dir,
            whisper_dir,
            avatar,
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
        self.avatar = avatar
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
            config: PlayerConfig,
            **kwargs
    ) -> np.ndarray:
        # audio_feature_batch = pe(whisper_batch)
        # latent_batch = []
        # for face_img in face_img_batch:
        #     latent_batch.append(
        #         self.vae.get_latents_for_unet(face_img)
        #     )
        # latent_batch = latent_batch.to(dtype=unet.model.dtype)
        #
        # pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        # recon = vae.decode_latents(pred_latents)
        # for res_frame in recon:
        #     res_frame_list.append(res_frame)
        pass
