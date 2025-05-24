from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

import functools
from torchdiffeq import odeint

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, sfm_params, mask, n_timesteps, temperature=1.0, alpha=1.0, solver='euler', spks=None, cond=None):
        xp, tp, logsigma_p = sfm_params
        x0 = torch.randn_like(xp) * temperature
        sigma_p = torch.exp(logsigma_p)

        Delta = torch.clamp(alpha*(torch.sqrt(sigma_p) + (1 - self.sigma_min) * tp), min=1.)
        tp = alpha * tp / Delta
        sigma_p = alpha**2 * sigma_p / Delta**2
        xp = alpha * xp / Delta
        
        xp = torch.sqrt(torch.clamp((1 - (1 - self.sigma_min) * tp)**2 - sigma_p, min=0.)) * x0 + xp

        t_span = torch.linspace(tp.item(), 1, n_timesteps + 1, device=xp.device)
        estimator = functools.partial(self.estimator, mask=mask, spks=spks)
        trajectory = odeint(estimator, xp, t_span, method=solver, rtol=1e-5, atol=1e-5)

        return trajectory, tp.item(), torch.sqrt(sigma_p).item()

    def compute_loss(self, x1, mask, sfm_params, spks=None, cond=None):
        b, _, t = x1.shape
        xp, tp, logsigma_p = sfm_params

        th = torch.sum(xp.detach() * x1, dim=1) / (torch.sum(x1 * x1, dim=1) + 1e-8) # [b, seq]
        th = torch.sum(th, dim=1) / torch.sum(mask.squeeze(1), dim=1)
        th = th[:, None, None]
        #th = torch.clamp(th, min=0.)
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

        x0 = torch.randn_like(x1)
        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        xp = torch.sqrt(torch.clamp((1 - (1 - self.sigma_min) * th)**2 - sigma_h, min=0.)) * x0 + xp
        xt = (1 - t) * xp + t * (x1 + self.sigma_min * x0)
        ut = (x1 + self.sigma_min * x0 - xp) / (1 - th)

        xt = xt * mask
        ut = ut * mask
        t = (1 - th) * t + th

        vt = self.estimator(t.squeeze(), xt, mask, spks, cond)
        fm_loss = F.mse_loss(vt, ut, reduction="sum") / (torch.sum(mask) * ut.shape[1])

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
            "Delta": torch.mean(Delta)
        }


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=out_channel,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
