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
   
## Conclusion and Analysis

Model Architecture and Training:

Our Convolutional Neural Network (CNN), leveraging the robust VGG16 as a base model, embarked on a journey to recognize and classify movie genres based on poster images from our dataset. The model architecture was meticulously crafted, incorporating convolutional layers for feature extraction, max-pooling layers for dimensionality reduction, and dense layers for final classification. The inclusion of dropout layers and data augmentation aimed to mitigate overfitting, ensuring the model generalizes well to unseen data.
Performance Metrics:

    Training Accuracy: 40.53%
    Validation Accuracy: 42.18%
    Loss: 0.2098 (training), 0.2005 (validation)

Insights and Reflections:

- Accuracy Achievements:
    - The model, after 50 epochs of training, achieved an its higest accuracy of approximately 40.53% on the training data and 42.18% on the validation data at epoch 19.

- Loss Considerations:
    - The loss values, 0.2098 for training and 0.2005 for validation, indicate that the model was able to learn patterns and features from the dataset, minimizing the discrepancy between predicted and actual labels. The proximity of training and validation loss suggests a balanced model without significant overfitting or underfitting.

- Challenges and Limitations:
    - The task of accurately predicting movie genres from posters is inherently complex due to the subtle and often subjective visual cues associated with genres. Moreover, the multi-label nature of the problem, where a movie can belong to multiple genres, adds an additional layer of complexity and challenge.

- Future Enhancements:
    - While the model demonstrated a commendable ability to recognize and predict genres from movie posters, future iterations could benefit from further hyperparameter tuning, exploration of alternative pre-trained models, and potentially leveraging additional data sources (e.g., movie synopsis, director, cast) to enhance predictive capabilities.

Concluding Remarks:

In essence, our CNN, through its journey of learning from movie posters, demonstrated a promising ability to discern and predict movie genres, albeit with room for further enhancement and optimization. The experiences, both triumphs and challenges, encountered throughout this project have not only enriched our understanding of image classification with deep learning but have also paved the way for future explorations and improvements in leveraging CNNs for visual recognition tasks. The insights gleaned from this endeavor will undoubtedly inform and inspire our subsequent ventures into the realm of machine learning and artificial intelligence.