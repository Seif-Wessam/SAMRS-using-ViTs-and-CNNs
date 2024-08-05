# Image Segmentation Performance Comparison: ViTs vs. CNNs

This repository contains the implementation and results of a project comparing the performance of Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) on image segmentation tasks using the SAMRS dataset.

## Table of Contents
- [Introduction](#introduction)
- [Description](#description)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to compare the performance of ViTs and CNNs on remote sensing image segmentation. The SAMRS dataset, specifically designed for remote sensing applications, serves as the benchmark for this comparison.

## Description
This project evaluates and compares Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) on remote sensing image segmentation using the SAMRS dataset.

## Dataset
The SAMRS dataset is used in this project. It includes annotations and images for semantic segmentation tasks in the remote sensing domain. For more details, visit the [SAMRS GitHub page](https://github.com/ViTAE-Transformer/SAMRS).

## Model Architectures
### Vision Transformer (ViT)
The ViT model is implemented based on the architecture described in the [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer) repository.

### Convolutional Neural Network (CNN)
The CNN model used is a standard U-Net architecture, widely recognized for its effectiveness in image segmentation tasks.

## Installation
Clone this repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/ViT-vs-CNN-Segmentation.git
cd ViT-vs-CNN-Segmentation
pip install -r requirements.txt
```

## Usage
### Data Preparation
Ensure the SAMRS dataset is downloaded and properly organized into train, val, and test directories with corresponding images and labels.

## Training
Train both the ViT and CNN models using the following scripts:
### Train ViT model
python train_vit.py --data_path /path/to/SAMRS --output_dir /path/to/save/models

#### Train CNN model
python train_cnn.py --data_path /path/to/SAMRS --output_dir /path/to/save/models

## Evaluation
Evaluate the performance of the trained models using:
### Evaluate ViT model
python evaluate_vit.py --data_path /path/to/SAMRS --model_path /path/to/saved/vit_model

### Evaluate CNN model
python evaluate_cnn.py --data_path /path/to/SAMRS --model_path /path/to/saved/cnn_model

## Results
The performance of both models is evaluated using metrics such as mean Intersection over Union (mIoU) and Overall Accuracy (OA). The results and comparison will be documented in the results directory.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you have suggestions for improvements or find any bugs.

## License
This project is licensed under the MIT License. The SAMRS dataset is available for academic purposes under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.

## Citation
If you use the SAMRS dataset in your research, please cite the following paper, off of which this project is based:

@inproceedings{SAMRS,
 author = {Wang, Di and Zhang, Jing and Du, Bo and Xu, Minqiang and Liu, Lin and Tao, Dacheng and Zhang, Liangpei},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {8815--8827},
 title = {SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model},
 volume = {36},
 year = {2023}
}

For any questions or further information, feel free to contact seif.hamed@epfl.ch.









