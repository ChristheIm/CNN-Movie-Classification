# Movie Genre Classification by Convolutional Neural Network
## Project Overview

This project aims to classify the genre of movies based on their poster images using a Convolutional Neural Network (CNN). The movie genre classification is a multi-label classification problem since a movie can belong to more than one genre. The CNN model will analyze the input image (movie poster) and predict the probability of the movie belonging to various genres.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training the Model](#training-the-model)
6. [Model Evaluation](#model-evaluation)
7. [Usage](#usage)

## Installation and Setup

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- Scikit-learn

### Installation

1. Clone the repository:
   ```shell
   git clone [repository_link]

2. Navigate to the project directory:
3. ```shell
   cd [project_directory]

# Data Collection and Preprocessing
### Data Collection
- The dataset consists of movie posters along with their respective genre labels.
- Data can be obtained from various movie databases like IMDB, TMDB, etc.
### Data Preprocessing
- Resize images to a standard size (e.g., 224x224 pixels).
- Normalize pixel values.
- Encode genre labels using MultiLabelBinarizer.

# Model Architecture
### CNN Model
- The CNN model is designed to extract features from the movie posters.
- It consists of convolutional layers, pooling layers, and fully connected layers. <br>
- Utilize activation functions, dropout, and batch normalization to enhance performance.

### Output Layer
- The output layer uses a sigmoid activation function to handle multi-label classification. <br>
- The model predicts the probability of the movie belonging to each genre. <br>

### Training the Model
- Split the data into training, validation, and test sets.
- Define a loss function suitable for multi-label classification (e.g., binary crossentropy).
- Choose an optimizer and set a learning rate.
- Train the model using the training data and validate it using the validation data.
- Save the model after training.

# Model Evaluation
- Evaluate the model using the test set.
- Utilize metrics like accuracy, precision, recall, F1-score, and ROC-AUC for evaluation.
- Analyze the confusion matrix to understand misclassification.

# Usage

### Predicting Movie Genre
Load the trained model.
Preprocess the input movie poster.
Use the model to predict the genre probabilities.
Decode the predicted probabilities to genre labels.
   
