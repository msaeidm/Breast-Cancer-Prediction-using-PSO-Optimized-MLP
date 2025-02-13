# Design Document for PSO-Optimized MLP for Breast Cancer Prediction

## Overview

This project aims to predict breast cancer using patient data with an MLP neural network. The key innovation is to use Particle Swarm Optimization (PSO) to optimize the neural network's parameters, thereby increasing prediction accuracy. The pipeline comprises:

1. **Data Preprocessing:**
   - Load raw patient data (574 samples, 32 features).
   - Handle missing values (e.g., using maximum frequency imputation).
   - Normalize features using Min/Max normalization.

2. **MLP Neural Network:**
   - Define an MLP with one input layer, one or more hidden layers, and one output layer.
   - Train using backpropagation.

3. **PSO Optimization:**
   - Use PSO to optimize MLP parameters (e.g., number of hidden neurons and hidden layers).
   - Each particle represents a candidate setting (n, m) and is evaluated by the MLP prediction accuracy.
   - Update particle velocities and positions according to PSO equations.

4. **Baseline Comparison:**
   - Train baseline classifiers such as Naïve Bayes, Decision Tree, and Nearest Neighbor.
   - Compare performance using accuracy, sensitivity, specificity, precision, and F-measure.

## Workflow Diagram

1. Data Preprocessing → 2. Train MLP Model → 3. Optimize using PSO → 4. Train Baselines → 5. Evaluate & Compare Results

## Pseudocode for PSO Optimization

