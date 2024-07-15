# London Weather Prediction: A Deep Learning Approach with GRU-based RNNs

This project focuses on predicting daily maximum, mean, and minimum temperatures in London using a deep learning model based on Gated Recurrent Units (GRUs). The project encompasses data acquisition, preprocessing, model definition, training, evaluation, and visualization of prediction results.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Goal](#project-goal)
3. [Methodology](#methodology)
    - [Data Acquisition and Exploration](#data-acquisition-and-exploration)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Definition and Architecture](#model-definition-and-architecture)
    - [Model Training](#model-training)
    - [Model Testing and Evaluation](#model-testing-and-evaluation)
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [Dependencies](#dependencies)

## Introduction

This project leverages deep learning techniques to predict weather patterns, specifically daily temperatures, in London. Utilizing a GRU-based RNN, the model is trained on historical weather data from 1979 to 2021 provided by the European Climate Assessment (ECA).

## Project Goal

The primary objective is to develop a robust and accurate deep learning model to forecast daily temperature parameters (maximum, mean, and minimum) using historical weather data. The model aims to predict the subsequent day's temperature using a configurable window of 50 days of historical data.

## Methodology

### Data Acquisition and Exploration

1. **Data Source**: Historical weather data from the European Climate Assessment (ECA) covering 1979-2021.
2. **Exploration**: Visualization techniques such as line plots and histograms to understand data distributions and identify seasonal trends.

### Data Preprocessing

1. **Handling Missing Values**: Techniques like forward/backward filling, interpolation, and median imputation.
2. **Scaling**: Robust scaling to handle outliers and normalize data.
3. **Feature Engineering**: Introducing cyclic features (sine and cosine transformations) to capture seasonal patterns.
4. **Data Preparation**: Creating input-output pairs and splitting data into training, validation, and testing subsets.

### Model Definition and Architecture

1. **Model Choice**: GRU-based RNN architecture due to its efficiency in handling long-term dependencies.
2. **Architecture**:
    - GRU layer for processing input sequences.
    - Fully connected layers with ReLU activation.
    - Linear transformation for dimensionality reduction and final predictions.

### Model Training

1. **Loss Function**: Mean Squared Error (MSE).
2. **Training Process**: Iterative parameter updates over 10 epochs with monitoring of training and validation performance to avoid overfitting.

### Model Testing and Evaluation

1. **Evaluation Metrics**: Average test loss and visual assessment of predictions against actual values.
2. **Performance**: Commendable prediction accuracy with the mean temperature exhibiting the highest fidelity.

## Results

The GRU-based model demonstrated effective learning of temporal dependencies, resulting in accurate predictions of daily maximum, mean, and minimum temperatures. Visual comparisons of predicted and actual temperatures showcased the model's proficiency.

## Conclusion

The developed deep learning model successfully predicts daily temperature trends in London, highlighting its potential for practical weather forecasting applications.


## Dependencies

- Python 3.8+
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch
- Jupyter Notebook

Ensure all dependencies are installed using the provided `requirements.txt` file.

## Authors

- Sliman Jammal
- Muhammad Eid

For any questions or feedback, feel free to reach out to Sliman Jammal at [sliman.jammal99@gmail.com](mailto:sliman.jammal99@gmail.com).
