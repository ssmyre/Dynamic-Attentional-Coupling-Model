import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from scipy import signal
import torch
from skimage.transform import resize
import os
import pickle

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


          
class Multimodal_Datasets_Split(Dataset):
    def __init__(self, dataset_path, data='mouse_vocal', split_type='train'):
        super(Multimodal_Datasets_Split, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        if split_type=='train':
            idx=np.arange(0,18452,1)
        elif split_type=='test':
            idx=np.arange(18542,20758,1)
        elif split_type=='valid':
            idx=np.arange(20758,23064,1)

        # These are torch tensors
        self.vision = dataset['vision'] 
        # new_shape = (self.vision.shape[0],240, 320)
        # # Resize the image array to the new dimensions (N, 240,320)
        # self.vision = resize(self.vision, new_shape)
        # self.vision -=np.mean(self.vision, axis =0)

        self.vision = torch.tensor(self.vision[idx,:,:].astype(np.float32), device='cpu').detach() #because visual frames dataset is very large, can not load onto GPU
        self.audio = dataset['audio'][idx,:,:].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio, device='cpu').detach()
        self.text = torch.tensor((np.zeros([self.audio.shape[0],1,1]).astype(np.float32)), device='cpu').detach()        
        
        # Note: this is STILL an numpy array
        self.meta = None

        self.n_modalities = 3 # vision/ text/ audio
        
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
       return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return self.audio.shape[0]
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        return X  

class Multimodal_Datasets_Split_Color(Dataset):
    def __init__(self, dataset_path, data='mouse_vocal', split_type='train'):
        super(Multimodal_Datasets_Split_Color, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))

        # size= int(dataset['vision'].shape[0])
        # trainingSize= int(size * 0.8)
        # testingSize = size - trainingSize

        if split_type=='train':
            idx=np.arange(0,18452,1)
        elif split_type=='test':
            idx=np.arange(18452,20758,1)
        elif split_type=='valid':
            idx=np.arange(20758,23061,1)

        # These are torch tensors
        self.vision = dataset['vision'] 
        # new_shape = (self.vision.shape[0],240, 320)
        # # Resize the image array to the new dimensions (N, 240,320)
        # self.vision = resize(self.vision, new_shape)
        # mframes = np.mean(self.vision, axis = 0)
        # output_dir = '/home/ssmyre/pearson_isilon/scott/MMT/data/Xu_Data/20230924_AngelHair/'
        # np.save(os.path.join(output_dir, 'All_Video','PG2camera23456_10'+split_type+ '_mean_vision.npy'), mframes)
        mframes = np.load('/home/ssmyre/duhs/Scott/Bird/grn213/Video/test_mean_vision.npy')
        self.vision -= mframes

        self.vision = torch.tensor(self.vision[idx,:,:,:].astype(np.float32), device='cpu').detach() #because visual frames dataset is very large, can not load onto GPU
        self.audio = dataset['audio'][idx,:,:].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio, device='cpu').detach()
        self.text = torch.tensor((np.zeros([self.audio.shape[0],1,1]).astype(np.float32)), device='cpu').detach()        
        
        # Note: this is STILL an numpy array
        self.meta = None

        self.n_modalities = 3 # vision/ text/ audio
        
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
       return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return self.audio.shape[0]
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        return X  


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset, batch_size):
        super(Multimodal_Datasets, self).__init__()
       
        _, self.text, self.audio, self.vision=dataset[:]
        # These are torch tensors
        self.text = torch.split(self.text, batch_size)
        self.audio = torch.split(self.audio, batch_size)
        self.vision = torch.split(self.vision, batch_size)

        self.text = self.text[0:-1]
        self.audio = self.audio[0:-1]
        self.vision = self.vision[0:-1]
        self.n_modalities = 3 # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return 1, 128, 240
    def get_dim(self):
        return 1, 128, 320
    def get_lbl_info(self):
        # return number_of_labels, label_dim
       return self.text.shape[1], self.text.shape[2]
    def __len__(self):
        return len(self.vision)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        return X 