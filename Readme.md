# Emotion Recognition from Faces (FER-2013)

This project implements a Deep Learning model (Convolutional Neural Network - CNN) to classify human emotions from facial images. It uses the FER-2013 dataset, a common benchmark for facial expression recognition.

## Table of Contents

### 1. Introduction

### 2. Dataset

### 3. Configuration

### 4. Data Loading & Preprocessing

### 5. Model Architecture

### 6. Training Process

### 7. Evaluation & Results

### 8. Saving Model & History

### 9. Class Distribution Analysis

### 10. Real-time Emotion Detector & Analysis

### 11. Usage

### 12. Dependencies


## 1. Introduction

The goal of this project is to build a machine learning model capable of identifying 7 basic human emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) from grayscale facial images. This capability has wide applications in human-computer interaction, sentiment analysis, and mental health monitoring.

## 2. Dataset

The FER-2013 (Facial Expression Recognition 2013) Dataset is used for this project.

**Source**: Commonly available on Kaggle and other academic platforms.

**Format**: Consists of 48x48 pixel grayscale images of faces.

**Structure**: Images are organized into train and test subdirectories, with further subfolders for each of the 7 emotion classes (e.g., train/happy/, test/angry/).

**Labels**: Emotion labels are inferred directly from the subdirectory names.

## 3. Configuration

Key parameters for the project are defined at the top of the script:

**base_dataset_path**: Root directory where the FER-2013 dataset (train and test folders) is located. (MUST BE UPDATED LOCALLY)

**IMG_HEIGHT, IMG_WIDTH**: Image dimensions (48x48 pixels).

**BATCH_SIZE**: Number of images processed per training step (e.g., 32 or 64).

**NUM_CLASSES:** Number of emotion categories (7).

**EPOCHS:** Maximum number of training iterations over the entire dataset.

**SEED:** Random seed for reproducibility.

**emotion_labels:** A dictionary mapping integer labels (0-6) to human-readable emotion names.

## 4. Data Loading & Preprocessing

The tf.keras.utils.image_dataset_from_directory utility is used for efficient data loading.

**Automatic Labeling:** Labels are inferred from folder names (labels='inferred', label_mode='int').

**Grayscale Conversion:** Images are explicitly loaded as grayscale (color_mode='grayscale') to match the model's input expectations.

**Resizing:** All images are resized to IMG_HEIGHT x IMG_WIDTH.

**Pixel Normalization:** Pixel values (0-255) are scaled to a 0-1 floating-point range (image = tf.cast(image, tf.float32) / 255.0). This is crucial for neural network training efficiency.

**Performance Optimization:** Datasets are cache()d and prefetch()ed to optimize data pipeline performance during training.

**Train/Test Split:** The dataset's inherent train and test folder structure is used for data splitting.

## 5. Model Architecture

A Convolutional Neural Network (CNN) is employed for emotion recognition, designed to effectively learn spatial features from images.

**Type:** keras.Sequential model.

**Input Layer:** Expects grayscale images of shape (IMG_HEIGHT, IMG_WIDTH, 1).

**Convolutional Blocks:** Three blocks, each consisting of:

**layers.Conv2D:** Applies filters (32, 64, 128) to learn features.

**layers.BatchNormalization():** Stabilizes training and speeds up convergence.

**layers.MaxPooling2D():** Reduces spatial dimensions.

**layers.Dropout():** Regularization (25% rate) to prevent overfitting.

**Flatten Layer:** Converts the 3D output of convolutional layers into a 1D vector.

**Dense (Fully Connected) Layers:**

- A hidden layers.Dense(256, activation='relu') layer for high-level feature learning.

- Includes BatchNormalization() and Dropout(0.5) for regularization.

- Output Layer: layers.Dense(NUM_CLASSES, activation='softmax'). Outputs probabilities for each of the 7 emotion classes.

## 6. Training Process

The model is compiled and trained using standard deep learning practices.

**Optimizer:** Adam optimizer with a default learning rate (0.001).

**Loss Function:** SparseCategoricalCrossentropy, suitable for integer labels.

**Metrics:** accuracy is monitored during training.

**Training Loop:** model.fit() trains the model for a specified number of EPOCHS (100).

**Validation:** test_ds is used as validation_data to monitor performance on unseen data during training, crucial for detecting overfitting.

## 7. Evaluation & Results

After training, the model's performance is evaluated on the dedicated test set.

**Test Loss & Accuracy:** model.evaluate(test_ds) provides the final loss and accuracy on the unseen test data.

**Overfitting Diagnosis:** The script prints the Final Training Accuracy and Final Validation Accuracy. A significant gap (e.g., Training Acc. >> Validation Acc.) indicates overfitting, meaning the model memorized training data but doesn't generalize well. Strategies like Early Stopping are crucial to combat this.

## 8. Saving Model & History

The trained model and training history are saved for future use and reproducibility.

**Model Save:** The entire trained Keras model is saved to emotion_recognition_model.h5 using model.save().

**Label Map Save:** The emotion_labels dictionary is saved to emotion_labels.json so that the numerical predictions can be mapped back to human-readable emotion names during inference.

**Training History Save:** The history object from model.fit() is converted to a Pandas DataFrame and saved as a timestamped CSV file in a history/ subdirectory (e.g., history/training_history_YYYYMMDD_HHMMSS.csv).

## 9. Class Distribution Analysis

The script includes a section to analyze and visualize the distribution of emotion classes in the training data.

**Process:** Extracts all labels from train_ds, counts occurrences using Pandas value_counts(), and maps numerical labels to emotion names.

**Visualization:** A seaborn.barplot is used to display the count of images for each emotion, visually highlighting class imbalance (e.g., more 'Happy' images, fewer 'Disgust' images). This insight is critical for understanding potential model biases and planning strategies like weighted loss or data augmentation.

## 10. Real-time Emotion Detector & Analysis

This project is structured into two separate Python scripts to optimize workflow and resource usage:

**train_emotion_model.py (Training Script):** This script is dedicated solely to loading the dataset, building, training, and saving the deep learning model. Running this is computationally intensive and results in the emotion_recognition_modelfer.h5 file. This separation avoids the need to retrain the model every time you want to use it for prediction.

**realtime_emotion_detector.py (Inference & Analysis Script):** This script loads the pre-trained emotion_recognition_modelfer.h5 model and emotion_labels.json map. It then performs two main tasks:

**Model Analysis:** It conducts a detailed analysis of the model's performance on the test set, including generating a Confusion Matrix and a Classification Report. This provides insights into which emotions the model confuses and its precision/recall for each class.

**Real-time Webcam Detection:** It deploys the model for live facial emotion recognition using a webcam, detecting faces, predicting emotions, and displaying results visually.

This modular approach ensures that the resource-intensive training process is run only when needed, and the lighter-weight inference and analysis can be performed efficiently.

## 11. Usage

### **To run this project:**

**Download and Extract Dataset:** Download the FER-2013 dataset. Ensure the train and test folders (containing emotion subfolders like angry, disgust, etc.) are located correctly relative to your base_dataset_path (e.g., ./data/train, ./data/test).

**Update Configuration:** Modify the base_dataset_path variable in the train_emotion_model.py script to point to your dataset's location.

**Install Dependencies:**

pip install tensorflow matplotlib numpy pandas scikit-learn opencv-python

Run the Training Script: Execute the training Python script (e.g., train_emotion_model.py) from your terminal or IDE.

python train_emotion_model.py

This script will train the model, evaluate it, and save the emotion_recognition_modelfer.h5 and emotion_labels.json files.

Run the Real-time Detector Script: Once the model is saved, execute the real-time detector Python script (e.g., realtime_emotion_detector.py) from your terminal or IDE.

python realtime_emotion_detector.py

## 12. Dependencies

tensorflow

matplotlib

numpy

pandas

scikit-learn

opencv-python (for real-time detection)