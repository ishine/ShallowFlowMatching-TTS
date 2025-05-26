import os
import random

import json
import torch
from torch.utils.data import Dataset

from text import cleaned_text_to_sequence

def intersperse(lst: list, item: int):
    """
    putting a blank token between any two input tokens to improve pronunciation
    see https://github.com/jaywalnut310/glow-tts/issues/43 for more details
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
    
class StableDataset(Dataset):
    def __init__(self, filelist_path, hop_length):
        self.filelist_path = filelist_path     
        self.hop_length = hop_length  
        self.spk_map = torch.load("spk_map.pt")
        
        self._load_filelist(filelist_path)

    def _load_filelist(self, filelist_path):
        filelist, lengths = [], []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                filelist.append((line['mel_path'], line['phone']))
                lengths.append(line['mel_length'])
            
        self.filelist = filelist
        self.lengths = lengths # length is used for DistributedBucketSampler
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        mel_path, phone = self.filelist[idx]
        spk = self.spk_map[mel_path.split("/")[-1].split("_")[1].strip("p")]
        mel = torch.load(mel_path, map_location='cpu', weights_only=True)
        phone = torch.tensor(intersperse(cleaned_text_to_sequence(phone), 0), dtype=torch.long)
        return mel, phone, spk
    
def collate_fn(batch):
    texts = [item[1] for item in batch]
    mels = [item[0] for item in batch]
    spks = [item[2] for item in batch]
    
    text_lengths = torch.tensor([text.size(-1) for text in texts], dtype=torch.long)
    mel_lengths = torch.tensor([mel.size(-1) for mel in mels], dtype=torch.long)
    spks = torch.tensor([spk for spk in spks], dtype=torch.long)
    
    # pad to the same length
    texts_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(texts), padding=0)
    mels_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels), padding=0)

    return texts_padded, text_lengths, mels_padded, mel_lengths, spks

# random slice mel for reference encoder to prevent overfitting
def random_slice_tensor(x: torch.Tensor):
    length = x.size(-1)
    if length < 12:
        return x 
    segmnt_size = random.randint(length // 12, length // 3)
    start = random.randint(0, length - segmnt_size)
    return x[..., start : start + segmnt_size]