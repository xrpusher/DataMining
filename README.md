# Project Overview

This project implements a machine learning framework using PyTorch to create a neural network model. It includes both a fixed model (Target Network) and an inverse model (Inverse MLP), alongside an Expectation-Maximization algorithm for Gaussian Mixture Models (GMM).

# File Descriptions
1. Target Network Implementation
2. 
File Purpose:
Defines the architecture of a fixed Multi-Layer Perceptron (MLP) network, referred to as Target.
Key Components:
Architecture:
A neural network with four layers, utilizing Tanh activation functions.
Weight Initialization: Weights are initialized uniformly between 0 and π.
Input/Output:
Takes 2-dimensional input and produces a single output.

3. Inverse MLP Implementation
   
File Purpose:
Implements an inverse neural network model (InverseMLP) to predict the original inputs based on the outputs from the Target network.
Key Components:
Architecture: Consists of multiple layers, including dropout layers for regularization.
Training: The model is trained using Mean Squared Error (MSE) loss with early stopping based on validation loss.

4. Gaussian Mixture Model (GMM) Implementation
   
File Purpose:
Implements the Expectation-Maximization (EM) algorithm for fitting a Gaussian Mixture Model.
Key Components:
Multivariate Gaussian Density Function: Computes the probability density function for multivariate normal distribution.
E-Step and M-Step: Performs the expectation and maximization steps to fit the model parameters.
Visualization:
Plots the Gaussian distributions and clusters based on the model parameters.

5. Data Generation and Visualization
   
File Purpose:
Generates synthetic data and visualizes results from both the Target network and the Inverse MLP.
Key Components:
Data Generation:
Creates input data based on a normal distribution.
Training and Validation:
Splits the dataset into training and validation sets, and normalizes the data.
Plots:
Displays training and validation loss over epochs, actual vs. predicted values, and residual distributions.

6. Utility Functions

File Purpose:
Contains helper functions for data processing, visualization, and model evaluation.
Key Components:
Residuals Distribution:
Plots the distribution of prediction errors.
Ideal Prediction Line:
Adds a line representing ideal predictions on scatter plots.
Installation Requirements
To run this project, ensure you have the following libraries installed:

# Usage Instructions
To train the models, run the respective scripts for the Target and Inverse MLP implementations.
Visualizations will automatically generate once the models have been trained, displaying various insights about model performance.
Conclusion
This project serves as an illustrative example of using neural networks in a supervised learning context, along with clustering methods using Gaussian Mixture Models. The architecture and implementation strategies demonstrated can be further expanded for more complex applications.

There is also a problem with a proof - task 4
