"""
BYOL Trainer
"""
from torch.utils.tensorboard import SummaryWriter
from dataloader import BYOLDataloader
from torchvision import models
from pix_byol import PixelCL
import torch.nn as nn
# import horovod.torch as hvd
from tqdm import tqdm
import torch



# Summery Writer #########################################################
writer = SummaryWriter(comment='Metrics')

# Config #################################################################
batch_size = 4
no_epoches = 100
device = 'cuda'

# Backbone ###############################################################
resnet = models.resnet34(pretrained=True)


# Dataset ################################################################
dataset = BYOLDataloader(data_path=r'D:\Datasets\image\areal\Semantic segmentation dataset\Tile 1')
trainloader = torch.utils.data.DataLoader(dataset,
                                          shuffle=True,
                                          batch_size=batch_size)
# Initialization #########################################################
def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)
resnet.apply(initialize_weights)
##########################################################################

# Define BYOL ############################################################
learner = PixelCL(
    resnet,
    image_size = 256,
    hidden_layer_pixel = 'layer4',  # leads to output of 8x8 feature map for pixel-level learning
    hidden_layer_instance = -2,     # leads to output for instance-level learning
    projection_size = 256,          # size of projection output, 256 was used in the paper
    projection_hidden_size = 2048,  # size of projection hidden dimension, paper used 2048
    moving_average_decay = 0.99,    # exponential moving average decay of target encoder
    ppm_num_layers = 1,             # number of layers for transform function in the pixel propagation module, 1 was optimal
    ppm_gamma = 2,                  # sharpness of the similarity in the pixel propagation module, already at optimal value of 2
    distance_thres = 0.7,           # ideal value is 0.7, as indicated in the paper, which makes the assumption of each feature map's pixel diagonal distance to be 1 (still unclear)
    similarity_temperature = 0.3,   # temperature for the cosine similarity for the pixel contrastive loss
    alpha = 1.,                      # weight of the pixel propagation loss (pixpro) vs pixel CL loss
    use_pixpro = True,               # do pixel pro instead of pixel contrast loss, defaults to pixpro, since it is the best one
    cutout_ratio_range = (0.6, 0.8)  # a random ratio is selected from this range for the random cutout
).to(device)

# Optimizer ##############################################################
opt = torch.optim.Adam(learner.parameters(),
                       lr=0.0001,
                       weight_decay=15e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                       T_max=len(trainloader),
                                                       eta_min=0,
                                                        last_epoch=-1)


# Training Loop ##########################################################
loop = tqdm(range(0, no_epoches))
batch_counter = 0
for ep in loop:
    epoch_loss = []
    epoch_ccloss = []
    for param_group in opt.param_groups:
        current_lr = param_group['lr']
    for idx, batch in enumerate(trainloader):
        images = batch.to(device)
        loss, ccloss, pix_loss, instance_loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()

        loop.set_postfix({
                          'Epoch': ep,
                          'Batch_Loss': loss.item(),
                          'current_lr': current_lr,
                          'batch id': " [{h1}/{h2}]".format(h1=idx, h2=len(trainloader)),
                          'ccloss': ccloss.item(),
                          'instance_loss': instance_loss.item(),
                          'pix loss': pix_loss.item()
                          })
        batch_counter += 1
        learner.update_moving_average()
        writer.add_scalar('Loss/Batch', loss, batch_counter)
        epoch_loss.append(loss.item())
        epoch_ccloss.append(ccloss.item())
    scheduler.step()

    if (ep + 1) % 5 == 0:
        torch.save(resnet.state_dict(), 'ckpt/{}.pth'.format(str(ep + 1)))
    writer.add_scalar('Lr/Epoch', current_lr, ep)
    writer.add_scalar('Loss/Epoch', sum(epoch_loss) / len(epoch_loss), ep)
    writer.add_scalar('ccloss/Epoch', sum(epoch_ccloss) / len(epoch_loss), ep)
########################## ****** End ****** ###############################