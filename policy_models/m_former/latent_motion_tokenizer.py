import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict
import numpy as np
import cv2
import collections.abc


class LatentMotionTokenizer(nn.Module):
    def __init__(
        self,
        m_former,
        vector_quantizer,
        m_former_hidden_size=768,
        codebook_embed_dim=32,
    ):
        super().__init__()

        self.m_former = m_former
        self.vector_quantizer = vector_quantizer

        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, m_former_hidden_size),
            nn.Tanh(),
            nn.Linear(m_former_hidden_size, codebook_embed_dim),
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, m_former_hidden_size),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        cond_pixel_values,
        target_pixel_values,
        return_motion_token_ids_only=False,
    ):
        batch = cond_pixel_values.shape[0]

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_pixel_values,
            target_hidden_states=target_pixel_values,
        ).last_hidden_state[:, :query_num]

        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)

        if return_motion_token_ids_only:
            return indices

        latent_motion_tokens_up = self.vq_up_resampler(quant)

        return latent_motion_tokens_up, indices, commit_loss