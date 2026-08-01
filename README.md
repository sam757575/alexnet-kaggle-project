# AlexNet Kaggle Project

This project implements the AlexNet deep learning architecture using a dataset from Kaggle. The goal is to train a model that can classify images effectively using the AlexNet architecture.

## Project Structure

- **data/**
  - **raw/**: Contains the raw dataset files downloaded from Kaggle.
  - **processed/**: Holds the processed dataset files ready for training.
  
- **notebooks/**
  - **exploration.ipynb**: Jupyter notebook for exploratory data analysis (EDA) on the dataset, including visualizations and insights.

- **src/**
  - **data/**
    - **dataset.py**: Contains the `Dataset` class for loading and preprocessing the dataset.
  - **models/**
    - **alexnet.py**: Defines the `AlexNet` class, implementing the AlexNet architecture.
  - **training/**
    - **train.py**: Contains the training script with functions for training and evaluating the model.
  - **utils/**
    - **helpers.py**: Includes utility functions for data augmentation and model saving.

- **tests/**
  - **test_models.py**: Contains unit tests for the model implementation.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **pyproject.toml**: Used for project configuration, specifying build system requirements and project metadata.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd alexnet-kaggle-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `data/raw/` directory.

4. Run the exploratory data analysis notebook:
   ```
   jupyter notebook notebooks/exploration.ipynb
   ```

5. Train the model using the training script:
   ```
   python src/training/train.py
   ```

## Overview of AlexNet

AlexNet is a convolutional neural network architecture that was designed to classify images into various categories. It consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers. This architecture significantly improved the performance of image classification tasks and laid the groundwork for future developments in deep learning.

## Usage

After training the model, you can use it to make predictions on new images. Refer to the `train.py` script for details on how to evaluate the model and make predictions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
