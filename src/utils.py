import torch
import os
from MMT.src.dataset import Multimodal_Datasets
from torch.utils.data import DataLoader

def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset+split)
    data= torch.load(data_path) 
    # if not os.path.exists(data_path):
    #     print(f"  - Creating new {split} data")
    #     data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
    #     torch.save(data, data_path)
    # else:
    #     print(f"  - Found cached {split} data")
    #     data = torch.load(data_path)
    return data

# def get_dataloader(data, batch_size=4, max_length=256,
#         stride=128, shuffle=True, drop_last=True, num_workers=0):
#     Loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
#     dataset = MouseDatasetV1(Loader, max_length, stride)

#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,
#         num_workers=num_workers 
#     )
    
#     return dataloader

def get_dataloader(data):
    data_loader = Multimodal_Datasets(data)
    return data_loader

def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model

# def save_state(filename):
#     """Save all the model parameters to the given file."""
#     layers = self._get_layers()
#     state = {}
#     for layer_name in layers:
#         state[layer_name] = layers[layer_name].state_dict()
#     state['optimizer_state'] = self.optimizer.state_dict()
#     state['loss'] = self.loss
#     state['z_dim'] = self.z_dim
#     state['epoch'] = self.epoch
#     state['lr'] = self.lr
#     state['save_dir'] = self.save_dir
#     filename = os.path.join(self.save_dir, filename)
#     torch.save(state, filename)    

def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')

def save_model_epochs(args, model, name=''):
    name = save_load_name(args, name)
    file_dir = os.path.join(args.root,args.save_dir)
    filename = os.path.join(file_dir,name)
    print(filename)
    torch.save(model, f'{filename}.pt')

def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model
