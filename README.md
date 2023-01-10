# Semantic segmentation of aerial imagery using U-Net 

PyTorch implementation of UNet for semantic segmentation of aerial imagery
1. This repository enables training UNet with various encoders like ResNet18, ResNet34, etc.
2. Uses a compound (Cross-Entropy + Jaccard loss) loss to train the network.
3. You can quickly use a custom dataset to train the model.
4. Contains a self-supervised method to train network encoder on unlabeled data (Upcoming task).
<html>
<p align="center">
   <img width="600" height="600" src="https://raw.githubusercontent.com/sabadijou/aerial_imagery_semantic_segmentation/main/demo/result.png">

</p>
</html>
## Upcoming Task
1. Complete the self-supervised part of the repository and train the encoder on our unlabeled dataset.
2. Implement dataloaders and evaluation metrics for FloodNet, and EarthVision challenges.
3. Add Distributed Data-Parallel strategy to the repository to enable multi-GPUs training.

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
   #### Note: this repository is still developing, and this dataset is used for testing the model. Our model could achieve 81.0% segmentation accuracy on this dataset.
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
  ## Training 
  The following command is prepared as an example for training the network. You can customize the parameters to train the default version.
   #### Note that this repository still not supports multi-GPUs training.
  ```Shell
  python train.py --dataset_path Semantic_segmentation_dataset --encoder resnet34 --encoder_weights imagenet --gpu_id 0 --gpus 1
  ```

  ## Demo
1. Inference Demo with a Pre-trained model.
   You can download our pretrain weights from [here](https://drive.google.com/file/d/1PkwkcttiLyyAkt45SuGWBmR5nUu9_CDf/view?usp=share_link) and customize the following command to run the demo 
  ```Shell
  python demo.py --checkpoint checkpoints/best.pth --image_path demo/sample_1.jpg --fname demo/result_1.jpg --gpu_id 0
  ```
