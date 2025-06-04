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
        
        self._load_filelist(filelist_path)

        token1 = torch.load("CosyVoice-libritts-data/train-clean-100/utt2speech_token.pt")
        token2 = torch.load("CosyVoice-libritts-data/train-clean-360/utt2speech_token.pt")
        token3 = torch.load("CosyVoice-libritts-data/train-other-500/utt2speech_token.pt")
        self.tokens = token1 | token2 | token3

        utt0 = torch.load("CosyVoice-libritts-data/train-clean-100/utt2embedding.pt")
        utt1 = torch.load("CosyVoice-libritts-data/train-clean-360/utt2embedding.pt")
        utt2 = torch.load("CosyVoice-libritts-data/train-other-500/utt2embedding.pt")
        self.utts = utt0 | utt1 | utt2

    def _load_filelist(self, filelist_path):
        filelist, lengths = [], []
        with open(filelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                mel_path = line['mel_path']
                file = mel_path.split("/")[-1].strip(".pt").split("_", 1)[-1]
                filelist.append((mel_path, file))
                lengths.append(line['mel_length'])
            
        self.filelist = filelist
        self.lengths = lengths # length is used for DistributedBucketSampler
    
    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        mel_path, file = self.filelist[idx]
        mel = torch.load(mel_path, map_location='cpu', weights_only=True)
        embed = self.utts[file]
        token = self.tokens[file]
        return mel, token, embed
    
def collate_fn(batch):
    mels = [item[0] for item in batch]
    tokens = [item[1] for item in batch]
    embeds = [item[2] for item in batch]
    
    token_lengths = torch.tensor([len(token) for token in tokens], dtype=torch.long)
    mel_lengths = torch.tensor([mel.size(-1) for mel in mels], dtype=torch.long)
    
    # pad to the same length
    tokens_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(tokens), padding=0)
    mels_padded = torch.nested.to_padded_tensor(torch.nested.nested_tensor(mels), padding=0)
    embeds = torch.tensor(embeds)

    return tokens_padded, token_lengths, mels_padded, mel_lengths, embeds
