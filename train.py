import torch.backends.cudnn as cudnn
import configs.kaggle_dataset as cfg
from utils.trainer import Trainer
from models.model import Model
import argparse


def main():
    args = parse_args()
    cfg.gpus = len(args.gpus)
    if args.gpu_id == '-1':
        cfg.device = 'cpu'
    else:
        cfg.device = 'cuda:{}'.format(int(args.gpu_id[0]))
    cfg.unet['backbone'] = args.encoder
    cfg.unet['pretrained'] = args.encoder_weights

    cfg.dataset_path = args.dataset_path

    cudnn.benchmark = True
    cudnn.fastest = True

    unet = Model(encoder_network=cfg.unet['backbone'],
                 encoder_depth=cfg.unet['encoder_depth'],
                 input_ch=cfg.unet['input_ch'],
                 out_channels=cfg.num_classes).to(cfg.device)

    trainer_module = Trainer(cfg, model=unet)
    trainer_module.run_train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--dataset_path', type=str, default='path_to_dataset',
        help='path_to_dataset')
    parser.add_argument(
        '--encoder', default='resnet34',
        help='encoder network = ')
    parser.add_argument(
        '--encoder_weights', default='imagenet',
        help='whether to initialize encoder from the checkpoint. for example "imagenet"')

    parser.add_argument('--gpus', nargs='+', type=int, default='0')
    parser.add_argument('--gpu_id', nargs='+', type=int, default='-1')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()