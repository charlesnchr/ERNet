import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageOps
import random
import pickle

import numpy as np

from skimage import io

def PSNR(I0,I1):
    MSE = torch.mean( (I0-I1)**2 )
    PSNR = 20*torch.log10(1/torch.sqrt(MSE))
    return PSNR


#normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                               std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(48),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

normalize2 = transforms.Normalize(mean = [0.69747254,0.53480325,0.68800158], std = [0.23605522,0.27857294,0.21456957])
unnormalize2 = transforms.Normalize(mean = [-2.9547, -1.9198, -3.20643], std = [4.2363, 3.58972, 4.66049])


toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      


def GetDataloaders(opt):
    if opt.dataset.lower() == 'pickledataset': 
        dataloader = load_GenericPickle_dataset(opt.root,'train',opt)
        validloader = load_GenericPickle_dataset(opt.root,'valid',opt)
    else:
        print('unknown dataset')
        return None,None
    return dataloader, validloader


    

class GenericPickleDataset(Dataset):

    def __init__(self, root, category,opt): # highres images not currently scaled, optdefault
        # fid = open(root + '/DIV2K_' + category + '_HR_%dx%d.pkl' % (imageSize,imageSize),'rb')
        # self.images = glob.glob(root + '/DIV2K_' + category + '_HR_bins%dx%d/*' % (opt.imageSize,opt.imageSize))
        
        self.images = glob.glob(root + '/*.npy')

        random.seed(1234)
        random.shuffle(self.images)

        if category == 'train':
            self.images = self.images[:opt.ntrain]
        else:
            self.images = self.images[-opt.ntest:]

        self.trans = transforms.Compose([transforms.ToPILImage(),transforms.Resize(opt.imageSize),transforms.ToTensor()])
        
        self.nch = opt.nch_in
        self.len = len(self.images)
        self.category = category

    def __getitem__(self, index):

        lq, hq = pickle.load(open(self.images[index], 'rb'))
        lq, hq = toTensor(lq), toTensor(hq)

        # multi-image input?
        if lq.shape[0] > self.nch:
            lq = lq[lq.shape[0] // 2].unsqueeze(0)
            hq = hq[hq.shape[0] // 2].unsqueeze(0)
        
        # rotate and flip?
        if self.category == 'train':
            if random.random() > 0.5:
                lq = lq.permute(0, 2, 1)
                hq = hq.permute(0, 2, 1)
            if random.random() > 0.5:
                lq = torch.flip(lq, [1])
                hq = torch.flip(hq, [1])
            if random.random() > 0.5:
                lq = torch.flip(lq, [2])
                hq = torch.flip(hq, [2])

        
        return lq, hq, hq # hq, lq, lq

    def __len__(self):
        return self.len       


def load_GenericPickle_dataset(root, category,opt):

    dataset = GenericPickleDataset(root, category, opt)
        
    if category == 'train':
        dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    else:
        dataloader = DataLoader(dataset, batch_size=opt.batchSize_test, shuffle=False, num_workers=0)
    return dataloader



