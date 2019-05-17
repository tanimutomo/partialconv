import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob


class Places2(Dataset):
    def __init__(self, img_root, mask_root, img_transform, mask_transform, data='train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        # get the list of image paths
        if data == 'train':
            self.paths = glob('{}/data_large/**/*.jpg'.format(img_root), recursive=True)
        else:
            self.paths = glob('{}/{}_large/*.jpg'.format(img_root, split))
        
        self.mask_paths = glob('{}/*.jpg'.format(mask_root))
        self.N_mask = len(self.mask_paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img - self.img_transform(img.convert('RGB'))
        mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = self.mask_transform(mask.convert('RGB'))
        return img * mask, mask, img

