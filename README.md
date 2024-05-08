# Handwritten Digit Classification using Convolutional Neural Networks (CNNs)

This repository contains code for training and evaluating a Convolutional Neural Network (CNN) model to classify handwritten digits from the MNIST dataset.

## Overview

The MNIST dataset is a widely used benchmark dataset in the field of machine learning and computer vision. It consists of 28x28 grayscale images of handwritten digits (0 to 9) along with their corresponding labels. The goal of this project is to develop a CNN model capable of accurately classifying these digits.

## Requirements

- Python (>=3.6)
- TensorFlow (>=2.0)
- NumPy
- Matplotlib (for visualization, optional)

## Usage

1. Clone the repository:
``` https://github.com/<username>/TNSDC-Generative-AI-Naan-Mudhalvan.git ```
2. Navigate to the project directory:
``` cd TNSDC-Generative-AI-Naan-Mudhalvan ```
3. Run the training script to train the model
```Image Classification using Convolutional Neural Networks (CNNs).ipynb```


This will train the CNN model on the MNIST dataset, evaluate its performance, and display sample predictions.

## Model Architecture

The CNN model architecture used for this task consists of several convolutional layers followed by max-pooling layers for feature extraction. The extracted features are then passed through fully connected layers for classification. The model is compiled with the Adam optimizer and trained using sparse categorical cross-entropy loss.

## Results

After training the model for a fixed number of epochs, the following results were obtained:
- Training accuracy: [Training Accuracy]
- Test accuracy: [Test Accuracy]

## Acknowledgments

- The code in this repository is based on concepts learned from various online tutorials and resources.
- Credits to the creators of the MNIST dataset for providing the benchmark dataset.
