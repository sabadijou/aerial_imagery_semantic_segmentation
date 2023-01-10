# Semantic segmentation of aerial imagery using U-Net 

PyTorch implementation of UNet for semantic segmentation of aerial imagery
1. This repository enables training UNet with various encoders like ResNet18, ResNet34, etc.
2. Uses a compound (Cross-Entropy + Jaccard loss) loss to train the network.
3. You can quickly use a custom dataset to train this repository.
4. Contains a self-supervised method to train network encoder on unlabeled data (Upcoming task).


## Upcoming Task
1. Complete the self-supervised part of the repository and train the encoder on our unlabeled dataset.
2. Implement dataloaders and evaluation metrics for FloodNet, and EarthVision challenges.

## Get started
1. Clone the repository
    ```
    git clone https://github.com/sabadijou/aerial_imagery_semantic_segmentation.git
    ```
    We call this directory as `$RESA_ROOT`

2. Create an environment and activate it (We've used conda. but it is optional)

    ```Shell
    conda create -n aiss python=3.9 -y
    conda activate aiss
    ```

3. Install dependencies

    ```Shell
    # Install pytorch firstly, the cudatoolkit version should be same in your system. (you can also use pip to install pytorch and torchvision)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # Install following libraries
    pip install opencv-python
    pip install numpy
    pip install matplotlib
    pip install segmentation_models_pytorch
    ```
4. Download and extract dataset
   Download and extract this [Kaggle Dataset](https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery)
   #### Note: this repository is still developing, and this dataset is used for testing the repository. Our model could achieve 81.0 segmentation accuracy on this dataset.
   ```
   semantic-segmentation-of-aerial-imagery/
                  ├── Tile 1
                        ├── images
                           ├── image_part_001.jpg
                           ├── image_part_002.jpg
                           ├── ...
                        └── masks
                           ├── image_part_001.png
                           ├── image_part_002.png
                           ├── ...
                  ├── Tile 2
                  ├── .
                  ├── .
                  └── Tile 9
                  └── classes
   
   ```
  ## Get started
  1. Simply open train.py in a python editor and customize the hyperparameters section.
  ```Shell
  batch_size = 32
  num_workers = 32
  using_pixpro = True   # True if using SSL for using backbone to implement segmentation tasks,
                        # False if using SSL for using backbone to implement classification tasks,
  num_of_gpus = torch.cuda.device_count()
  image_folder_path = r'image_folder_path'
  # Define the backbone
  backbone = models.resnet34(pretrained=True)
  hidden_layer_pixel = 'layer4'
  ```
  2. Run train.py

  ## Acknowledgement
<!--ts-->
* [open-mmlab/mmselfsup](https://github.com/open-mmlab/mmselfsup)
* [lucidrains/pixel-level-contrastive-learning](https://github.com/lucidrains/pixel-level-contrastive-learning)
<!--te-->
