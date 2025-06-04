import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_encoder import TextEncoder
from models.flow_matching import CFMDecoder
from utils.mask import sequence_mask
from models.diffusion_transformer import DiTConVBlock

# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/matcha_tts.py
class StableTTS(nn.Module):
    def __init__(self, n_vocab, mel_channels, hidden_channels, filter_channels, n_heads, n_enc_layers, n_dec_layers, kernel_size, p_dropout, gin_channels):
        super().__init__()

        self.n_vocab = n_vocab
        self.mel_channels = mel_channels

        self.encoder = TextEncoder(n_vocab, mel_channels, hidden_channels, filter_channels, n_heads, n_enc_layers, kernel_size, p_dropout, gin_channels)
        self.decoder = CFMDecoder(mel_channels, mel_channels, hidden_channels, mel_channels, filter_channels, n_heads, n_dec_layers, kernel_size, p_dropout, gin_channels)
        
        # uncondition input for cfg
        self.fake_speaker = nn.Parameter(torch.zeros(1, gin_channels))
        #self.fake_content = nn.Parameter(torch.zeros(1, mel_channels, 1))
        
        self.cfg_dropout = 0.2

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, c, n_timesteps, temperature=1.0, alpha=1.0, length_scale=1.0, solver=None, cfg=1.0):

        x, x_mask = self.encoder(x, c, x_lengths)

        y_lengths = torch.round(x_lengths / 50 * 44100 / 512 * length_scale).long()
        y_max_length = y_lengths.max()
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)

        x = F.interpolate(x, size=y_max_length, mode='linear')
        mu, sfm_params = self.encoder.forward_smooth(x, y_mask)
        encoder_outputs = mu[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        if cfg == 1.0:
            decoder_outputs, tp, sigma_p = self.decoder(sfm_params, y_mask, n_timesteps, temperature, alpha, c, solver)
        else:
            cfg_kwargs = {'fake_speaker': self.fake_speaker, 'cfg_strength': cfg}
            decoder_outputs, tp, sigma_p = self.decoder(sfm_params, y_mask, n_timesteps, temperature, alpha, c, solver, cfg_kwargs)
            
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
        }, tp, sigma_p

    def forward(self, x, x_lengths, y, y_lengths, c):

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        y_mask = sequence_mask(y_lengths, y.size(2)).unsqueeze(1).to(y.dtype)
        cfg_mask = torch.rand(y.size(0), 1, device=y.device) > self.cfg_dropout
        
        x, x_mask = self.encoder(x, c, x_lengths)

        x = F.interpolate(x, size=y_lengths.max(), mode='linear')
        mu, sfm_params = self.encoder.forward_smooth(x, y_mask)

        # Compute loss of the decoder
        c = c * cfg_mask + ~cfg_mask * self.fake_speaker.repeat(x.size(0), 1)
        loss_dict, value_dict = self.decoder.compute_loss(y, y_mask, sfm_params, c)

        coarse_loss = F.mse_loss(mu, y, reduction="none")
        coarse_loss = torch.sum(coarse_loss) / (torch.sum(y_mask) * self.mel_channels)

        loss_dict["coarse_loss"] = coarse_loss

        return loss_dict, value_dict