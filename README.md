# MNIST Classifier with Training Visualization

This project implements a simple neural network using PyTorch to classify handwritten digits from the MNIST dataset. It includes real-time visualization of the training progress.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Features
- Simple feedforward neural network for MNIST digit classification
- Training and testing on the MNIST dataset
- Real-time visualization of training progress
- Jupyter notebook integration for interactive development

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- jupyter

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:avvem/MNIST-classifier.git
   cd MNIST-classifier
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install torch torchvision matplotlib jupyter
   ```

## Usage

### Using Jupyter Notebook in VS Code:

1. Open the project in VS Code.
2. Open the command palette (Cmd+Shift+P on Mac or Ctrl+Shift+P on Windows/Linux).
3. Type "Jupyter: Create New Blank Notebook" and select it.
4. Copy the content of the `mnist_classifier.py` into cells in the new notebook.
5. Run the cells to train the model and see the visualizations.

### Using Python Script:

1. Open a terminal in the project directory.
2. Run the script:
   ```
   python mnist_classifier.py
   ```
3. The training progress will be printed in the console, and visualizations will be saved as 'training_progress.png' after each epoch.

## Project Structure

- `mnist_classifier.py`: Main Python script containing the model definition, training loop, and visualization code.
- `MNIST_Classifier.ipynb`: Jupyter notebook version of the classifier (if created).
- `mnist_model_with_visualization.pth`: Saved model weights after training.
- `training_progress.png`: Visualization of training progress (updated after each epoch).

## Model Architecture

The model is a simple feedforward neural network with the following structure:
- Input layer: 784 neurons (28x28 flattened image)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit) with log softmax activation

## Training Process

The model is trained for 10 epochs using the Adam optimizer and Negative Log Likelihood Loss. After each epoch, the training loss, training accuracy, and test accuracy are calculated and visualized.

## Visualization

The training progress is visualized in two plots:
1. Training Loss over Epochs
2. Training and Test Accuracy over Epochs

These plots are updated after each epoch and saved as 'training_progress.png'.

## Results

After training, the final model accuracy on the test set is printed. The trained model is saved as 'mnist_model_with_visualization.pth'.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
