# U-Net Image Segmentation

U-Net image segmentation project! In this project, I built a U-Net, a type of Convolutional Neural Network (CNN) designed for quick and precise image segmentation. I used it to predict a label for every single pixel in an image from a self-driving car dataset.

## Project Overview

This project focuses on semantic image segmentation, a technique similar to object detection but with a key difference: instead of labeling objects with bounding boxes, semantic image segmentation predicts a precise mask for each object in the image by labeling each pixel with its corresponding class. This is crucial for applications like self-driving cars, which require a pixel-perfect understanding of their environment to make safe and accurate decisions.

### Key Objectives

1. **Build a U-Net:** Construct a U-Net architecture from scratch.
2. **Differentiate CNN and U-Net:** Understand the differences between a regular CNN and a U-Net.
3. **Semantic Image Segmentation:** Implement and apply semantic image segmentation on the CARLA self-driving car dataset.
4. **Sparse Categorical Crossentropy:** Utilize sparse categorical crossentropy for pixel-wise prediction.

## Getting Started

### Prerequisites

To get started with this project, the following libraries are required:

- TensorFlow
- Keras
- NumPy
- Matplotlib

### Dataset

I used the CARLA self-driving car dataset for this project. Ensure the dataset is downloaded and properly organized in your working directory.

### U-Net Architecture

The U-Net architecture consists of an encoder (downsampling path) and a decoder (upsampling path), connected by a bottleneck. Here's a high-level overview of the architecture:

1. **Encoder:** The encoder is a typical CNN that extracts features from the input image.
2. **Bottleneck:** The bottleneck layer captures the most crucial features.
3. **Decoder:** The decoder uses transposed convolutions to upsample the features back to the original image size, enabling pixel-wise classification.

## Implementation Steps

1. **Import Libraries:** Load necessary libraries such as TensorFlow, Keras, NumPy, and Matplotlib.
2. **Define U-Net Model:** Construct the U-Net model with an encoder, bottleneck, and decoder.
3. **Compile and Train the Model:** Compile the model using an optimizer and loss function, then train it on the dataset.
4. **Evaluate and Visualize Results:** Evaluate the model's performance on validation data and visualize the predicted masks.

## Results

After training the U-Net model, I evaluated its performance on the validation set. The model achieved promising results, accurately predicting pixel-wise masks for various objects in the images.
![download (12)](https://github.com/jaliyanimanthako/U-Net-Image-Segmentation-Cousera-Guided-Project-/assets/161110418/bb557623-b067-43f5-839e-a5269b28962e)

![image](https://github.com/jaliyanimanthako/U-Net-Image-Segmentation-Cousera-Guided-Project-/assets/161110418/40915fba-9a2e-45f4-979e-1339993a9f41)



## Acknowledgements

This project is inspired by the Deep Learning Specialization by deeplearning.ai on Coursera.

- [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)
- [Deep Learning Specialization](https://www.deeplearning.ai/program/deep-learning-specialization/)
- [Deep Learning with PyTorch : Image Segmentation](https://www.coursera.org/account/accomplishments/verify/27ULZG8NP4MD)
