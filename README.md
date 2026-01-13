# CIFAR-10 Image Classification with CNN using TensorFlow/Keras

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using TensorFlow and Keras.

## Project Overview

The goal of this project is to create a CNN model that can classify images into one of 10 categories from the CIFAR-10 dataset. The notebook includes the following steps:

1. **Data Loading and Preprocessing**: Loading the CIFAR-10 dataset and normalizing the image pixel values.
2. **Model Definition**: Constructing a simple CNN architecture using `tf.keras.Sequential`.
3. **Model Compilation and Training**: Compiling the model with the Adam optimizer and `sparse_categorical_crossentropy` loss, and training it for one epoch (extendable).
4. **Model Evaluation**: Evaluating the model's performance on the test set.
5. **Model Summary**: Displaying the architecture and parameter count of the model.
6. **Prediction Examples**: Demonstrating how to make predictions on new images.
7. **Intermediate Activation Visualization**: Visualizing the output of intermediate convolutional layers to understand what features the network is learning.
8. **Training History Visualization**: Plotting training and validation accuracy and loss over epochs.

## Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into:

* 50,000 training images
* 10,000 test images

## Setup and Usage

To run this notebook, you can use either Google Colab or a local Python environment.

### Google Colab (Recommended)

1. Open the notebook in [Google Colab](https://colab.research.google.com/drive/1XAFzKztysjDaLci_J_-ZWsEDwFYdWN_M?usp=sharing).
2. Run all cells sequentially (`Runtime -> Run all`).


## Model Architecture

The CNN model architecture used in this notebook consists of the following layers:

1. `Conv2D` (32 filters, 3x3 kernel, ReLU activation, input shape 32x32x3)
2. `MaxPooling2D` (2x2 pool size)
3. `Conv2D` (64 filters, 3x3 kernel, ReLU activation)
4. `MaxPooling2D` (2x2 pool size)
5. `Flatten`
6. `Dense` (64 units, ReLU activation)
7. `Dense` (10 units, Softmax activation for classification)

## Results

After training for 1 epoch, the model achieves approximately the following results:

* **Training Accuracy**: ~63%
* **Test/Validation Accuracy**: ~65%

These results can be further improved by:

* Increasing the number of training epochs
* Experimenting with different model architectures
* Adding regularization (e.g., dropout)
* Using data augmentation techniques

## Visualizations

* **Intermediate Activations**: Visualizing the output of the first convolutional layer, showing how different filters respond to the input image.
* **Training Plots**: Graphs illustrating the training and validation accuracy and loss over epochs, helping to monitor model performance and detect overfitting or underfitting.
