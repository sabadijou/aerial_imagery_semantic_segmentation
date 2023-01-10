import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    def __init__(self, encoder_network='resnet34',
                 encoder_depth=5,
                 input_ch=3,
                 out_channels=1,
                 pretrained='imagenet'):

        super(Model, self).__init__()
        self.unet = smp.Unet(encoder_name=encoder_network,
                             encoder_depth=encoder_depth,
                             encoder_weights=pretrained,
                             in_channels=input_ch,
                             classes=out_channels)

    def forward(self, x):
        output = self.unet(x)
        return output