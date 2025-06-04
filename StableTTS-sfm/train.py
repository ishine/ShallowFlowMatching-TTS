import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from dataclasses import asdict

from datas.dataset import StableDataset, collate_fn
from datas.sampler import DistributedBucketSampler
from text import symbols
from config import MelConfig, ModelConfig, TrainConfig
from models.model import StableTTS

from utils.scheduler import get_cosine_schedule_with_warmup
from utils.load import continue_training

torch.backends.cudnn.benchmark = True
    

def _init_config(model_config: ModelConfig, mel_config: MelConfig, train_config: TrainConfig):
    if not os.path.exists(train_config.model_save_path):
        print(f'Creating {train_config.model_save_path}')
        os.makedirs(train_config.model_save_path, exist_ok=True)

def train():
    rank = 0
    local_rank = 0
    world_size = 1

    model_config = ModelConfig()
    mel_config = MelConfig()
    train_config = TrainConfig()
    
    _init_config(model_config, mel_config, train_config)
    
    model = StableTTS(len(symbols), mel_config.n_mels, **asdict(model_config)).to(local_rank)

    train_dataset = StableDataset(train_config.train_dataset_path, mel_config.hop_length)
    train_sampler = DistributedBucketSampler(train_dataset, train_config.batch_size, [32,300,400,500,600,700,800,900,1000], num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
    
    if rank == 0:
        writer = SummaryWriter(train_config.log_dir)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    steps_per_epoch = len(train_dataloader)
    num_training_steps = train_config.num_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * 0.1), num_training_steps=num_training_steps)
    scaler = GradScaler()

    # load latest checkpoints if possible
    current_epoch = continue_training(train_config.model_save_path, model, optimizer)

    model.train()
    steps = 0
    for epoch in range(current_epoch, train_config.num_epochs):  # loop over the train_dataset multiple times
        train_dataloader.batch_sampler.set_epoch(epoch)
        if rank == 0:
            dataloader = tqdm(train_dataloader)
        else:
            dataloader = train_dataloader
            
        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(local_rank, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, c = datas
            optimizer.zero_grad()
            with autocast():
                loss_dict, value_dict = model(x, x_lengths, y, y_lengths, c)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if train_config.grad_clip_thresh:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_thresh)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            steps += 1
            if rank == 0 and steps % train_config.log_interval_step == 0:
                for key in loss_dict.keys():
                    writer.add_scalar(f"training/{key}", loss_dict[key].item(), steps)
                for key in value_dict.keys():
                    writer.add_scalar(f"value/{key}", value_dict[key].item(), steps)
                writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)
            
        if rank == 0 and (epoch+1) % train_config.save_interval_epoch == 0:
            torch.save({
                'model': model.state_dict(),
                #'optimizer': optimizer.state_dict(),
                #'scaler': scaler.state_dict()
            }, os.path.join(train_config.model_save_path, f'checkpoint_{epoch}.pt'))
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

if __name__ == "__main__":
    import numpy as np 
    import random
    worker_seed = 1234
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    train()