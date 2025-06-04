from dataclasses import dataclass
from typing import Optional

@dataclass
class MelConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    f_min: float = 0.0
    f_max: float = None
    pad: int = 0
    n_mels: int = 128
    center: bool = False
    pad_mode: str = "reflect"
    mel_scale: str = "slaney"
    
    def __post_init__(self):
        if self.pad == 0:
            self.pad = (self.n_fft - self.hop_length) // 2
            
@dataclass
class ModelConfig:
    n_vocab: int = 4096
    mel_channels: int = 128
    hidden_channels: int = 256
    filter_channels: int = 1024
    n_heads: int = 4
    n_enc_layers: int = 6
    n_dec_layers: int = 6 
    kernel_size: int = 3
    p_dropout: int = 0.1
    gin_channels: int = 192
            
@dataclass
class TrainConfig:
    train_dataset_path: str = 'filelists/filelist.json'
    #test_dataset_path: str = 'filelists/filelist.json' # not used
    batch_size: int = 64
    learning_rate: float = 1e-4 # the same with CosyVoice
    grad_clip_thresh: Optional[float] = 1.0 # the same with CosyVoice
    num_epochs: int = 200 # the same with CosyVoice
    model_save_path: str = './checkpoints'
    log_dir: str = './runs'
    log_interval_step: int = 100
    save_interval_epoch: int = 20
    
@dataclass
class VocosConfig:
    input_channels: int = 128
    dim: int = 512
    intermediate_dim: int = 1536
    num_layers: int = 8