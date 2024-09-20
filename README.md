# Convolutional Neural Network (CNN) Project

 ## Introduction
 This project demonstrates the implementation of a Convolutional Neural Network (CNN) using PyTorch.
 CNNs are a class of deep learning models typically used for image classification, object detection, 
 and other tasks involving visual data.

# What is a CNN?
 A Convolutional Neural Network (CNN) is a type of artificial neural network designed to process 
 and recognize patterns in visual data. It is widely used in computer vision tasks such as:

 - Image Classification
 - Object Detection
 - Image Segmentation

# CNNs consist of several key layers:

 - Convolutional Layers: Extract features from the input images using filters.
 - Pooling Layers: Downsample the feature maps, reducing dimensionality and computation.
 - Fully Connected Layers: Perform the final classification based on the extracted features.
 - Activation Functions: Non-linear functions (like ReLU) applied to the outputs of layers.

# Dataset

 The dataset used in this project is:
 - **MNIST**: A dataset of handwritten digits with 60,000 training examples and 10,000 test examples, 
 each image being a grayscale image of size 28x28 pixels.

 Dataset Preparation:
 The dataset is downloaded using PyTorch's `torchvision.datasets` module.
 Data is preprocessed using transformations such as normalization and conversion to tensor format.

# GPU for PyTorch Notebook
## Overview
This Jupyter Notebook is designed to demonstrate the utilization of GPU with PyTorch for enhanced computational efficiency. It includes basic checks to verify GPU availability and operations to manipulate tensors on GPU.

## Features
 GPU Availability Check: Verifies if CUDA is available on the device and identifies the GPU device name.
 Memory Management: Includes examples of how to check and manage GPU memory allocation.
 Tensor Operations: Demonstrates how to create and operate on tensors directly on the GPU.
 Model Training: Includes a simple neural network model setup for the Iris dataset using PyTorch’s neural network modules.
 Data Handling: Illustrates how to load, split, and preprocess data using Pandas and Scikit-learn.
 Training Loop: Executes a training loop, displaying the optimization of loss over epochs, and measures the training time.
