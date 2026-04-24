# Machine Learning Assignment 2: CNN Interpretability with CAM

This project contains my submission for Assignment 2 in the Machine Learning course. The assignment investigates interpretability in a pretrained convolutional neural network using Class Activation Maps (CAMs).

A pretrained ResNet18 model with ImageNet weights is used together with LayerCAM from the `torch-cam` library. The goal is to examine which parts of an image contribute to the model's classification decisions.

## Contents

- `report.ipynb`  
  Main report notebook containing the analysis, visualisations, results, and conclusions.

- `visualiser.py`  
  Helper class used to load the model, preprocess images, generate CAM overlays, retrieve predictions, and compare layers.

- `imagenet_class_index.json`  
  ImageNet class index file used to map model output indices to human-readable class labels.

- `images/`  
  Image files used for the positive, negative, and out-of-model examples.

## What is analysed

The report analyses three ImageNet classes:

- `golden_retriever`
- `tabby`
- `sports_car`

For each class, a positive example and a negative example are examined using CAM attribution maps. The report also includes:

- analysis of top predictions/logits
- comparison of early, middle, and late network layers
- an example containing a class not present in the pretrained model

## Purpose

The purpose of the project is to explore how a pretrained CNN makes classification decisions and how its internal representations change across layers. The attribution maps are used to discuss what visual features the model appears to rely on, such as facial features, texture, object parts, and overall shape.

## Dependencies

Main libraries used:

- PyTorch
- torchvision
- torch-cam
- matplotlib
