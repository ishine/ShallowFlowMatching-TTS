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
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
from matcha.models.components.flow_matching import BASECFM
from matcha.models.components.decoder import SinusoidalPosEmb, TimestepEmbedding

import functools
from torchdiffeq import odeint


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, mu, sfm_params, mask, n_timesteps, temperature=1.0, alpha=1.0, solver='euler', spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        xp, tp, logsigma_p = sfm_params
        x0 = torch.randn_like(xp) * temperature
        sigma_p = torch.exp(logsigma_p)

        Delta = torch.clamp(alpha*(torch.sqrt(sigma_p) + (1 - self.sigma_min) * tp), min=1.)
        tp = alpha * tp / Delta
        sigma_p = alpha**2 * sigma_p / Delta**2
        xp = alpha * xp / Delta

        x = torch.sqrt(torch.clamp(((1 - (1 - self.sigma_min) * tp))**2 - sigma_p, min=0.)) * x0 + xp

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=xp.device, dtype=xp.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t_span = t_span * (1. - tp.item()) + tp.item()

        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)

        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        spks_in[0] = spks
        cond_in[0] = cond

        estimator = functools.partial(self.cfg_wrapper, mask=mask_in, c=[spks_in, cond_in])
        trajectory = odeint(estimator, x_in, t_span, method=solver, rtol=1e-5, atol=1e-5)

        return trajectory[-1].float(), tp.item(), torch.sqrt(sigma_p).item()

    def cfg_wrapper(self, t, x, mask, c):
        spks, cond = c
        dphi_dt = self.estimator.forward(
                x, mask,
                t.repeat(2),
                spks,
                cond
            )

        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0)//2, x.size(0)//2], dim=0)
        dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
        dphi_dt = dphi_dt.repeat(2, 1, 1)
        return dphi_dt

    def compute_loss(self, x1, mask, sfm_params, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape
        xp, tp, logsigma_p = sfm_params

        th = torch.sum(xp.detach() * x1, dim=1) / (torch.sum(x1 * x1, dim=1) + 1e-8) # [b, seq]
        th = torch.sum(th, dim=1) / torch.sum(mask.squeeze(1), dim=1)
        th = th[:, None, None]
        sigma_h = F.mse_loss(xp.detach(), th * x1, reduction="none")
        sigma_h = torch.sum(sigma_h, dim=[1,2], keepdim=True) / (torch.sum(mask, dim=[1,2], keepdim=True) * x1.shape[1])

        Delta = torch.clamp(torch.sqrt(sigma_h) + (1 - self.sigma_min) * th, min=1.)
        th = th / Delta
        sigma_h = sigma_h / Delta**2
        xp = xp / Delta

        mu_loss = F.mse_loss(xp, th * x1, reduction="none")
        mu_loss = torch.sum(mu_loss) / (torch.sum(mask) * x1.shape[1])

        t_loss = F.mse_loss(tp, th, reduction="mean")
        sigma_loss = F.mse_loss(logsigma_p, torch.log(sigma_h + 1e-8), reduction="mean")

        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        
        x0 = torch.randn_like(x1)
        xp = torch.sqrt(torch.clamp((1 - (1 - self.sigma_min) * th)**2 - sigma_h, min=0.)) * x0 + xp
        xt = (1 - t) * xp + t * (x1 + self.sigma_min * x0)
        ut = (x1 + self.sigma_min * x0 - xp) / (1 - th)

        xt = xt * mask
        ut = ut * mask

        #during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            #mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        t = t * (1. - th) + th
        vt = self.estimator(xt, mask, t.squeeze(), spks, cond)
        fm_loss = F.mse_loss(vt, ut, reduction="sum") / (torch.sum(mask) * vt.shape[1])

        return {
            "fm_loss": fm_loss,
            "mu_loss": mu_loss,
            "t_loss": t_loss,
            "sigma_loss": sigma_loss,
        }, {
            "sigma_h": torch.mean(sigma_h),
            "sigma_p": torch.mean(torch.exp(logsigma_p)),
            "th": torch.mean(th),
            "tp": torch.mean(tp),
            "Delta": torch.mean(Delta),
        }
