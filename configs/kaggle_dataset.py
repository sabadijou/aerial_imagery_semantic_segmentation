unet = dict(
    backbone='resnet34',
    pretrained='imagenet',
    encoder_depth=5,
    input_ch=3
)

optimizer = dict(
  type='Adam',
  lr=1e-3,
  momentum=0.9,
  weight_decay=1e-4,
)

epochs = 100
batch_size = 4
world_size = 1

scheduler = dict(
    step_size=1,
    gamma=0.5
    )

eval_ep = 6
save_ep = epochs

bg_weight = 0.4

img_norm = dict(
    mean=[103.939, 116.779, 123.68],
    std=[1., 1., 1.]
)

img_height = 512
img_width = 512


dataset_path = r'D:\Datasets\image\areal\Semantic_segmentation_dataset'
dataset = dict(
        train_test_split=0.1
)
device = 'cuda'
gpus = 0
workers = 1
num_classes = 6
current_epoch = 0