CNN-LSTM Model Training Pipeline
Overview
This project implements a training pipeline for a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) model using TensorFlow and Keras. The model is trained on a provided dataset using specified hyperparameters and tracked using MLFlow. Model selection is based on a specified metric.

Requirements
Python 3.x
TensorFlow
Keras
MLFlow
Other dependencies listed in requirements.txt
Usage
Ensure all dependencies are installed by running pip install -r requirements.txt.
Configure the dataset and model parameters in src/data.py and src/model.py.
Run python train.py to start the training process.
Check MLFlow tracking server for training logs and metrics.
Components
CNN-LSTM Model: Defined in src/model.py, this class encapsulates the architecture and training configuration of the CNN-LSTM model.
Dataset: Defined in src/data.py, this class handles data loading, preprocessing, and splitting into training and testing sets.
Training: Defined in train.py, this class orchestrates the training process, including model compilation, training loop, and model selection based on specified metrics.
MLFlowTracker: Defined in src/experiment_tracking.py, this class provides functionality for tracking experiments using MLFlow.
Customization
Modify the dataset and model parameters in src/data.py and src/model.py to suit your specific use case.
Adjust hyperparameters such as batch size, number of epochs, and learning rate in train.py as needed.
Note
Ensure that MLFlow tracking server is running and accessible to track the training experiments. You can customize MLFlow configurations in mlflow.yml.
