# Breast Cancer Classification with Neural Network (PyTorch)

This project builds and trains a fully-connected neural network with two hidden layers to classify breast cancer tumors as **malignant** or **benign** using the Breast Cancer Wisconsin dataset from scikit-learn.

The goal of this project is to demonstrate foundational machine learning and deep learning skills, including:
- Data preprocessing
- Neural network design
- Activation functions
- Loss calculation
- Backpropagation
- Model evaluation

This project is intended for both learning purposes and professional portfolio use.

---

## Model Architecture

A simple feedforward neural network (Multi-Layer Perceptron) with the following structure:

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
├── breast_cancer_nn.py   # Main training and evaluation script
└── README.md             # Project documentation
