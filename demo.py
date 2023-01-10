

import torch.backends.cudnn as cudnn
import configs.kaggle_dataset as cfg
import matplotlib.pyplot as plt
from models.model import Model
import numpy as np
import torch
import argparse
import cv2


def main():
    args = parse_args()
    if args.gpu_id == '-1':
        cfg.device = 'cpu'
    else:
        cfg.device = 'cuda:{}'.format(int(args.gpu_id[0]))

    image_path = args.image_path
    checkpoint_path = args.checkpoint
    cfg.unet['backbone'] = args.encoder
    cudnn.benchmark = True
    cudnn.fastest = True

    unet = Model(encoder_network=cfg.unet['backbone'],
                 encoder_depth=cfg.unet['encoder_depth'],
                 input_ch=cfg.unet['input_ch'],
                 out_channels=cfg.num_classes).to(cfg.device)
    unet.load_state_dict(torch.load(checkpoint_path))
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512)) / 255.0
    image = np.moveaxis(image, -1, 0)
    image = torch.tensor(image).float().to(cfg.device).unsqueeze(0)
    result = unet(image)
    result = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(result)
    plt.savefig(fname=args.fname)
    print('The result is saved in {}'.format(args.fname))


def parse_args():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--image_path', type=str, help='path to image')
    parser.add_argument('--checkpoint', default='checkpoints/best.pth', help='path to checkpoint')
    parser.add_argument('--encoder', default='resnet34', help='encoder network = ')
    parser.add_argument('--fname', default='demo/result.jpg', help='where to save result: "demo/result.jpg"')
    parser.add_argument('--gpu_id', nargs='+', type=int, default='-1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()