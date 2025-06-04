import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def continue_training(checkpoint_path, model: DDP, optimizer: optim.Optimizer) -> int:
    """load the latest checkpoints and optimizers"""
    model_dict = {}
    optimizer_dict = {}
    
    # globt all the checkpoints in the directory
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pt") and '_' in file:
            name, epoch_str = file.rsplit('_', 1)
            epoch = int(epoch_str.split('.')[0])
            
            if name.startswith("checkpoint"):
                model_dict[epoch] = file
            elif name.startswith("optimizer"):
                optimizer_dict[epoch] = file
    
    # get the largest epoch
    common_epochs = set(model_dict.keys()) & set(optimizer_dict.keys())
    if common_epochs:
        max_epoch = max(common_epochs)
        model_path = os.path.join(checkpoint_path, model_dict[max_epoch])
        optimizer_path = os.path.join(checkpoint_path, optimizer_dict[max_epoch])
        
        # load model and optimizer
        state_dict = torch.load(model_path, map_location='cpu')
        try:
            model.load_state_dict(state_dict)
        except:
            try:
                print("Loading into model failed, try loading into model.module...")
                model.module.load_state_dict(state_dict)
            except:
                raise RuntimeError("Both model and model.module failed to load. Stop continuing training.")

        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        
        print(f'resume model and optimizer from {max_epoch} epoch')
        return max_epoch + 1
    
    else:
        # load pretrained checkpoint
        if model_dict:
            model_path = os.path.join(checkpoint_path, model_dict[max(model_dict.keys())])
            model.module.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
            
        return 0