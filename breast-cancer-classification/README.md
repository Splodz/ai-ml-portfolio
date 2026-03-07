# Breast Cancer Classification with Neural Network (PyTorch)

This project uses a feedforward neural network built in PyTorch to classify breast cancer tumors as malignant or benign. It demonstrates data preprocessing, model training, and binary classification evaluation as part of an AI/ML portfolio.

## Overview

This project implements a neural network in PyTorch to perform binary classification on the Breast Cancer Wisconsin dataset from scikit-learn. The goal is to predict whether a tumor is malignant or benign based on diagnostic features.

The project demonstrates the core steps of a machine learning pipeline:

- loading and preprocessing structured medical data
- splitting the dataset into training and test sets
- building and training a neural network
- evaluating model performance on unseen data
---

## Model Architecture

A simple feedforward neural network (Multi-Layer Perceptron) with the following structure:

Input Layer (30 features)
        ↓
Hidden Layer (32 neurons, ReLU)
        ↓
Hidden Layer (16 neurons, ReLU)
        ↓
Output Layer (1 neuron, Sigmoid)

| Layer  | Type            | Size             | Activation |
|-------:|-----------------|------------------|------------|
| 1      | Fully Connected | input_dim → 32   | ReLU       |
| 2      | Fully Connected | 32 → 16          | ReLU       |
| Output | Fully Connected | 16 → 1           | Sigmoid    |

Sigmoid activation is used at the output layer to model the probability for binary classification.

---

## Training Details

- **Loss Function:** Binary Cross Entropy (`BCELoss`)
- **Optimizer:** Adam (learning rate = 0.001)
- **Epoch:** 100
- **Batching:** Full batch (for simplicity)
- **Metrics:** Training and test accuracy

The training loop includes:
- Forward pass
- Loss computation
- Backpropagation (`loss.backward()`)
- Weight updates (`optimizer.step()`)

---

## Results

After 100 epochs of training, the model typically achieves:

- **90–97% accuracy on the test set**

Exact results may vary slightly due to randomness in data splitting and weight initialization.

---

## Project Structure

```text
breast_cancer_nn/
│
├── breast_cancer_nn.py      # Main training and evaluation script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

## Installation

Clone the repository:

```bash
git clone https://github.com/Splodz/ai-ml-portfolio.git
cd ai-ml-portfolio/breast-cancer-classification

Install dependencies:

```bash
pip install -r requirements.txt

Run the model:

```bash
python breast_cancer_nn.py
