backbone = dict(
    type='ResNetWrapper',
    resnet='resnet50',
    pretrained=True,
    replace_stride_with_dilation=[False, True, True],
    out_conv=True,
    fea_stride=8,
)

optimizer = dict(
  type='Adam',
  lr=1e-3,
  momentum=0.9,
  weight_decay=1e-4,
)

epochs = 12
batch_size = 8
total_iter = (88880 // batch_size) * epochs
import math
scheduler = dict(
    type = 'LambdaLR',
    lr_lambda = lambda _iter : math.pow(1 - _iter/total_iter, 0.9)
)

eval_ep = 6
save_ep = epochs

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 288
img_width = 800
cut_height = 240

dataset_path = r'D:\Datasets\image\areal\Semantic segmentation dataset'
dataset = dict(
    train=dict(
        type='',
        img_path=dataset_path,
        data_list=''

)
)

workers = 1
num_classes = 6
ignore_label = 255
log_interval = 500