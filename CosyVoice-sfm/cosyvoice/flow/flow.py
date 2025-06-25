# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from omegaconf import DictConfig
from cosyvoice.utils.mask import make_pad_mask
from matcha.models.components.text_encoder import LayerNorm

class SFM(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, output_channels):
        super().__init__()
        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, output_channels, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_size: int = 80,
                 spk_embed_dim: int = 192,
                 output_type: str = "mel",
                 vocab_size: int = 4096,
                 input_frame_rate: int = 50,
                 only_mask_loss: bool = True,
                 encoder: torch.nn.Module = None,
                 length_regulator: torch.nn.Module = None,
                 sfm: torch.nn.Module = None,
                 decoder: torch.nn.Module = None,
                 decoder_conf: Dict = {'in_channels': 240, 'out_channel': 80, 'spk_emb_dim': 80, 'n_spks': 1,
                                       'cfm_params': DictConfig({'sigma_min': 1e-06, 'solver': 'euler', 't_scheduler': 'cosine',
                                                                 'training_cfg_rate': 0.2, 'inference_cfg_rate': 0.7, 'reg_loss_type': 'l1'}),
                                       'decoder_params': {'channels': [256, 256], 'dropout': 0.0, 'attention_head_dim': 64,
                                                          'n_blocks': 4, 'num_mid_blocks': 12, 'num_heads': 8, 'act_fn': 'gelu'}},
                 mel_feat_conf: Dict = {'n_fft': 1024, 'num_mels': 80, 'sampling_rate': 22050,
                                        'hop_size': 256, 'win_size': 1024, 'fmin': 0, 'fmax': 8000}):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size) # , 512
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size) # 192, 80
        self.token_proj = nn.Linear(input_size+spk_embed_dim+output_size, input_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size) # 512, 80
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.sfm = sfm
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device) # [b, token_s, 1]
        token = self.input_embedding(torch.clamp(token, min=0)) # [b,s,512]
        token = torch.concat([token, 
                              embedding.unsqueeze(1).expand(-1, token.shape[1], -1),
                              F.interpolate(conds.transpose(1, 2).contiguous(), size=token_len.max(), mode='linear').transpose(1, 2).contiguous()], dim=-1)
        token = self.token_proj(token) * mask

        # text encode
        feat_mask = (~make_pad_mask(feat_len)).float().unsqueeze(-1).to(device) # [b, feat_s, 1]
        
        h, _ = self.encoder(token, token_len) # [b,s,512]
        h, _ = self.length_regulator(h * mask, feat_len) # [b,s,512], [b]
        h = h * feat_mask

        coarse_mel = self.encoder_proj(h) * feat_mask
        feat = F.interpolate(feat.unsqueeze(dim=1), size=coarse_mel.shape[1:], mode="nearest").squeeze(dim=1)
        coarse_loss = F.mse_loss(coarse_mel, feat, reduction="sum") / (torch.sum(feat_mask) * feat.shape[-1])
        
        feat_mask = feat_mask.transpose(1, 2).contiguous()
        x = self.sfm(h.transpose(1, 2).contiguous(), feat_mask)
        tp = x[:, :1, :]
        logsigma_p = x[:, 1:2, :]
        xp = x[:, 2:, :]
        tp = F.sigmoid(tp) * feat_mask
        tp = torch.sum(tp, dim=-1, keepdim=True) / torch.sum(feat_mask, dim=-1, keepdim=True) #[b, 1, 1]
        logsigma_p = torch.sum(logsigma_p, dim=-1, keepdim=True) / torch.sum(feat_mask, dim=-1, keepdim=True) #[b, 1, 1]

        conds = conds.transpose(1, 2).contiguous()
        feat = feat.transpose(1, 2).contiguous()
        embedding = self.spk_embed_affine_layer(embedding)
        loss_dict, value_dict = self.decoder.compute_loss(
            feat,
            feat_mask,
            [xp, tp, logsigma_p],
            embedding,
            cond=conds
        )

        loss_dict["coarse_loss"] = coarse_loss
        loss = sum(loss_dict.values())
        return {"loss": loss} | loss_dict | value_dict

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  n_timesteps,
                  temperature,
                  alpha,
                  solver):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)

        # get conditions
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device)
        conds[:, :mel_len1] = prompt_feat

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0))
        token = torch.concat([token, 
                              embedding.unsqueeze(1).expand(-1, token.shape[1], -1),
                              F.interpolate(conds.transpose(1, 2).contiguous(), size=token_len.max(), mode='linear').transpose(1, 2).contiguous()], dim=-1)
        token = self.token_proj(token) * mask

        # text encode
        h, _ = self.encoder(token, token_len)
        h = h * mask
        
        feat_mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).float().unsqueeze(-1).to(token.device)
        h, _ = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)
        h = h * feat_mask
        coarse_mel = self.encoder_proj(h) * feat_mask

        feat_mask = feat_mask.transpose(1, 2).contiguous()
        x = self.sfm(h.transpose(1, 2).contiguous(), feat_mask)
        tp = x[:, :1, :]
        logsigma_p = x[:, 1:2, :]
        xp = x[:, 2:, :]
        tp = F.sigmoid(tp) * feat_mask
        tp = torch.sum(tp, dim=-1, keepdim=True) / torch.sum(feat_mask, dim=-1, keepdim=True) #[b, 1, 1]
        logsigma_p = torch.sum(logsigma_p, dim=-1, keepdim=True) / torch.sum(feat_mask, dim=-1, keepdim=True) #[b, 1, 1]

        embedding = self.spk_embed_affine_layer(embedding)
        conds = conds.transpose(1, 2).contiguous()

        feat, tp, sigma_p = self.decoder(
            mu=coarse_mel.transpose(1, 2).contiguous(),
            sfm_params=[xp, tp, logsigma_p],
            mask=feat_mask, 
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            temperature=temperature, 
            alpha=alpha,
            solver=solver,
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat, tp, sigma_p