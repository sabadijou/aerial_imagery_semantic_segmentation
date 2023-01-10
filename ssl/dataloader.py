import numpy
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
import torch
import os



def img_transformer():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(size=(192, 320))])
    return transform


class BYOLDataloader(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super(BYOLDataloader, self).__init__()
        self.data_path = data_path
        self.sample_path = os.listdir(self.data_path)
        self.transformer = img_transformer()

    def __getitem__(self, idx):

        # Load Sample #########################################
        try:
            sample = Image.open(os.path.join(self.data_path,
                                             self.sample_path[idx]), mode='r', formats=None)
            # sample = sample.thumbnail((720, 1280))
            sample = numpy.array(sample)
            sample = np.resize(sample, new_shape=(504, 1280, 3))
            H, W, _ = sample.shape
            H_cut = int(H * 0.3)
            sample = sample[H_cut:H, ...]
        except:
            sample = Image.open(os.path.join (self.data_path,
                                               self.sample_path[0]), mode='r', formats=None)
            # sample = sample.thumbnail((720, 1280))
            sample = numpy.array(sample)
            sample = np.resize(sample, new_shape=(504, 1280, 3))
            H, W, _ = sample.shape
            H_cut = int(H * 0.3)
            sample = sample[H_cut:H, ...]

        sample = self.transformer(sample)
        return sample

    def __len__(self):
        return len(self.sample_path)
