# MNIST Classifier

This repository contains a simple neural network model built with PyTorch to classify handwritten digits from the MNIST dataset. The model is trained on the MNIST dataset and achieves decent accuracy on digit classification tasks.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)
- [License](#license)

## Features
- Simple feedforward neural network architecture
- Training and testing on the MNIST dataset
- Uses PyTorch for model implementation
- Includes data normalization and transformation

## Requirements
- Python 3.6 or higher
- PyTorch
- torchvision
- matplotlib

## Installation

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone git@github.com:avvem/MNIST-classifier.git
cd MNIST-classifier
```

### 2. Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies for this project. You can create a virtual environment using venv:

```bash
python3 -m venv myenv
```

### 3. Activate the Virtual Environment
Activate the virtual environment:

**On macOS/Linux:**
```bash
source myenv/bin/activate
```

**On Windows:**
```bash
myenv\Scripts\activate
```

### 4. Install Required Packages
With the virtual environment activated, install the required packages:

```bash
pip3 install torch torchvision matplotlib
```

## Usage

### Running the Model
To train the model, simply run the following command:

```bash
python3 mnist_classifier.py
```

This will download the MNIST dataset (if not already downloaded), train the model for 5 epochs, and print the test accuracy.

## Model Training
The model is trained for 5 epochs using the Adam optimizer and negative log-likelihood loss. The training process includes the following steps:

1. Load the MNIST dataset.
2. Define a simple feedforward neural network.
3. Train the model using the training data.
4. Evaluate the model's accuracy on the test data.

## Results
After training, the model will print the test accuracy, which indicates how well the model performs on unseen data.

## License
This project is licensed under the MIT License - see the LICENSE file for details.