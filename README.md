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
