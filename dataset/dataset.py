'''
Source: https://colab.research.google.com/drive/1vYZYXDMfs9hK6KvXY3v1iTPiln9OeJZE?usp=sharing#scrollTo=R6nhF1um1VBZ

Modified by https://github.com/sabadijou
'''

import torchvision.transforms as transforms
from configs.kaggle_dataset import device
from scipy import ndimage
import torch.utils.data
from glob import glob
import numpy as np
import torch
import cv2


class ArealDataset(torch.utils.data.Dataset):
    def __init__(self, root, training):
        super(ArealDataset, self).__init__()
        self.root = root
        self.training = training
        self.transform = transforms.Compose([transforms.ColorJitter(.1, .1, .1, .1),
                                             transforms.GaussianBlur(3, sigma=(0.1, 2.0))])
        self.IMG_NAMES = sorted(glob(self.root + '/*/images/*.jpg'))

        self.BGR_classes = {'Water': [41, 169, 226],
                            'Land': [246, 41, 132],
                            'Road': [228, 193, 110],
                            'Building': [152, 16, 60],
                            'Vegetation': [58, 221, 254],
                            'Unlabeled': [155, 155, 155]}

        self.bin_classes = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']

    def __getitem__(self, idx):
        img_path = self.IMG_NAMES[idx]
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        cls_mask = np.zeros(mask.shape)
        cls_mask[mask == self.BGR_classes['Water']] = self.bin_classes.index('Water')
        cls_mask[mask == self.BGR_classes['Land']] = self.bin_classes.index('Land')
        cls_mask[mask == self.BGR_classes['Road']] = self.bin_classes.index('Road')
        cls_mask[mask == self.BGR_classes['Building']] = self.bin_classes.index('Building')
        cls_mask[mask == self.BGR_classes['Vegetation']] = self.bin_classes.index('Vegetation')
        cls_mask[mask == self.BGR_classes['Unlabeled']] = self.bin_classes.index('Unlabeled')
        cls_mask = cls_mask[:, :, 0]

        if self.training:
            if self.transform:
                image = transforms.functional.to_pil_image(image)
                image = self.transform(image)
                image = np.array(image)

            # 90 degree rotation
            if np.random.rand() < 0.5:
                angle = np.random.randint(4) * 90
                image = ndimage.rotate(image, angle, reshape=True)
                cls_mask = ndimage.rotate(cls_mask, angle, reshape=True)

            if np.random.rand() < 0.5:
                image = np.flip(image, 0)
                cls_mask = np.flip(cls_mask, 0)

            if np.random.rand() < 0.5:
                image = np.flip(image, 1)
                cls_mask = np.flip(cls_mask, 1)

        image = cv2.resize(image, (512, 512)) / 255.0
        cls_mask = cv2.resize(cls_mask, (512, 512))
        image = np.moveaxis(image, -1, 0)

        return torch.tensor(image).float().to(device), torch.tensor(cls_mask, dtype=torch.int64).to(device)

    def __len__(self):
        return len(self.IMG_NAMES)
