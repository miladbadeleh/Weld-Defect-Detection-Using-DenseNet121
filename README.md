# DenseNet121 for Weld Defect Classification

This repository contains a PyTorch implementation of the DenseNet121 model for classifying weld defects. The model is trained on a custom dataset of weld images to detect various types of weld defects.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction
The goal of this project is to classify weld defects using a deep learning model. We use the DenseNet121 architecture, which is a convolutional neural network known for its efficiency and accuracy in image classification tasks. The model is trained on a dataset of weld images and can classify them into different categories of defects.

## Dataset
The dataset consists of images of welds, categorized into different types of defects. The dataset is divided into three subsets:
- **Training set**: Used to train the model.
- **Validation set**: Used to tune the model and prevent overfitting.
- **Test set**: Used to evaluate the final model performance.

The images are resized to 300x30 pixels and normalized using the mean and standard deviation of the ImageNet dataset.

## Model Architecture
The model used in this project is DenseNet121, which is a pre-trained model from the `torchvision.models` library. The architecture of DenseNet121 is characterized by dense blocks where each layer is connected to every other layer in a feed-forward fashion. This helps in feature reuse and improves the flow of gradients through the network.

The final fully connected layer of the DenseNet121 model is modified to match the number of classes in our dataset. The model is then trained using the Adam optimizer and CrossEntropyLoss as the loss function.

## Training
The training process involves the following steps:
1. **Data Loading**: The dataset is loaded using `torchvision.datasets.ImageFolder` and transformed using `transforms.Compose`.
2. **Model Initialization**: The DenseNet121 model is loaded with pre-trained weights, and the final layer is modified to match the number of classes.
3. **Training Loop**: The model is trained for 10 epochs. During each epoch, the model is trained on the training set and validated on the validation set. The training and validation loss and accuracy are printed after each epoch.
4. **Model Saving**: After training, the model is saved to a file named `densenet121_weld_defects_model.pth`.

## Testing
The trained model is evaluated on the test set to measure its performance. The test loop calculates the loss and accuracy of the model on the test set and prints the results.

## Results
The model achieves the following performance metrics:
- **Training Accuracy**: ~91%
- **Validation Accuracy**: ~90%
- **Test Accuracy**: ~90.5%

These results indicate that the model is able to generalize well to unseen data and can effectively classify weld defects.

## Usage
To use this model, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/densenet121_weld_defects.git
   cd densenet121_weld_defects
