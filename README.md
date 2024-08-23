# Speech Emotion Recognition (SER) Using Machine Learning & LSTM

This repository contains a Speech Emotion Recognition (SER) system built using Long Short-Term Memory (LSTM) neural networks. The project also evaluates other classification models including SVM, KNN, Random Forest, and Decision Tree. The project utilizes four popular datasets: TESS, RAVDESS, CREMA-D, and SAVEE, to classify emotions from audio speech data. Feature extraction is performed using Mel-Frequency Cepstral Coefficients (MFCC), and the analysis is conducted in a Jupyter Notebook.

## Table of Contents
- [Introduction](#introduction)
- [Datasets](#datasets)
- [Feature Extraction](#feature-extraction)
- [Libraries Used](#libraries-used)
- [Model Architectures](#model-architectures)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Speech Emotion Recognition (SER) is a challenging task in the field of machine learning and artificial intelligence. The goal is to classify the emotional state of a speaker based on their speech. This project leverages LSTM networks to capture temporal dependencies in speech data for better emotion classification and compares performance with other models like SVM, KNN, Random Forest, and Decision Tree.

## Datasets
The project uses the following datasets:
- **TESS** (Toronto Emotional Speech Set)
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **SAVEE** (Surrey Audio-Visual Expressed Emotion)

These datasets contain audio recordings of actors portraying various emotions, such as happiness, sadness, anger, fear, and surprise.

## Feature Extraction
Feature extraction is performed using Mel-Frequency Cepstral Coefficients (MFCC). The Jupyter Notebook file `feature_extraction.ipynb` contains the code for preprocessing audio files and extracting MFCC features, which are then used for training and evaluating the classification models.

## Libraries Used
The following libraries are used in the project:

- **`pandas`**: For data manipulation and analysis.
- **`numpy`**: For numerical computing and array operations.
- **`os`**: For operating system functionalities, like file and directory management.
- **`seaborn`**: For data visualization based on Matplotlib.
- **`matplotlib.pyplot`**: For basic plotting and visualization.
- **`librosa`**: For audio and music analysis, including feature extraction like MFCCs.
- **`librosa.display`**: For displaying audio signals and spectrograms.
- **`sklearn.preprocessing.OneHotEncoder`**: For converting categorical data into a one-hot numeric array.
- **`sklearn.preprocessing.LabelEncoder`**: For encoding categorical labels as numbers.
- **`sklearn.model_selection.train_test_split`**: For splitting data into training and testing sets.
- **`sklearn.svm.SVC`**: For Support Vector Classification model.
- **`sklearn.neighbors.KNeighborsClassifier`**: For K-Nearest Neighbors classification.
- **`sklearn.ensemble.RandomForestClassifier`**: For Random Forest classification.
- **`sklearn.tree.DecisionTreeClassifier`**: For Decision Tree classification.
- **`sklearn.model_selection.GridSearchCV`**: For hyperparameter tuning using cross-validation.
- **`sklearn.metrics.accuracy_score`**: For measuring the accuracy of classification models.
- **`IPython.display.Audio`**: For playing audio in Jupyter notebooks.
- **`speech_recognition`**: For recognizing speech from audio files or microphone input.
- **`pyttsx3`**: For text-to-speech conversion.
- **`tempfile`**: For creating temporary files and directories.
- **`warnings`**: For issuing warnings to the user.

## Model Architectures
The project evaluates several classification models:
- LSTM: Long Short-Term Memory network for capturing temporal dependencies in speech.
- SVM : Support Vector Machine for classification.
- KNN : K-Nearest Neighbors for classification.
- Random Forest: Ensemble of decision trees for classification.
- Decision Tree: Single decision tree for classification.

## Installation
To set up the project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/speech-emotion-recognition.git
    cd speech-emotion-recognition
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download and organize datasets**:
    - Download TESS, RAVDESS, CREMA-D, and SAVEE datasets.
    - Place them in the `datasets/` directory with the following structure:
        ```
        datasets/
        ├── TESS/
        ├── RAVDESS/
        ├── CREMA-D/
        └── SAVEE/
        ```

## Usage
To work with the Jupyter Notebook and process the data:

1. **Preprocess the data and extract features**:
    - Open the Jupyter Notebook file `feature_extraction.ipynb`.
    - Execute the cells to preprocess the audio data and extract MFCC features.

2. **Train and evaluate the models**:
    - Run the relevant scripts to train and evaluate LSTM, SVM, KNN, Random Forest, and Decision Tree models.

3. **Predict emotion from a new audio file**:
    ```bash
    python predict.py --file path_to_audio.wav
    ```

## Results
The performance of the models is evaluated using metrics like accuracy, precision, recall, and F1-score. Results indicate the effectiveness of each model in capturing emotions in speech:

| Model             | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| LSTM Baseline     | 0.96%    | 0.96%     | 0.96%  | 0.96%    |
| SVM               | 0.85%    | 0.85%     | 0.85%  | 0.85%    |
| KNN               | 0.78%    | 0.79%     | 0.78%  | 0.78%    |
| Random Forest     | 0.85%    | 0.85%     | 0.85%  | 0.85%    |
| Decision Tree     | 0.75%    | 0.75%     | 0.75%  | 0.75%    |


## Contributing
Contributions are welcome! If you have ideas to improve this project, feel free to fork the repository and submit a pull request.


