# Job Postings Fraud Detection

This project provides a machine learning pipeline to detect fraudulent job postings using natural language processing (NLP) and structured data features. It includes scripts for data preprocessing, feature engineering, model training, evaluation, and prediction.

## Project Structure

```
job_postings.csv
rf_model_with_transformers.joblib
test_model.py
Test.csv
train_model.py
```

- **job_postings.csv**: Main dataset for training and evaluation.
- **Test.csv**: Additional dataset for testing or validation.
- **train_model.py**: Script for data preprocessing, feature engineering, model training, and saving the trained model.
- **test_model.py**: Script for loading the trained model and evaluating it on a dataset.
- **rf_model_with_transformers.joblib**: Saved Random Forest model and transformers (TF-IDF, scaler, encoder).

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- nltk
- wordcloud
- matplotlib
- joblib

Install dependencies with:

```sh
pip install pandas numpy scikit-learn imbalanced-learn nltk wordcloud matplotlib joblib
```

## NLTK Data

The scripts will attempt to download required NLTK resources (`punkt`, `stopwords`) automatically.

## Usage

### 1. Train the Model

Run the training script to preprocess data, engineer features, train a Random Forest classifier, and save the model:

```sh
python train_model.py
```

This will:
- Clean and preprocess the data from `job_postings.csv`
- Generate a word cloud visualization
- Engineer features (TF-IDF, categorical encoding, scaling)
- Train a Random Forest classifier with SMOTE oversampling
- Save the trained model and transformers to `rf_model_with_transformers.joblib`

### 2. Evaluate the Model

To evaluate the trained model on a dataset (e.g., `job_postings.csv` or `Test.csv`):

```sh
python test_model.py
```

This will:
- Load the trained model and transformers
- Preprocess and engineer features from the specified dataset
- Predict and print the F1 score and classification report

You can specify a different dataset by editing the `main` function call in `test_model.py`.

## Main Functions

- `train_model.main`: Entry point for training and saving the model.
- `test_model.load_and_evaluate`: Loads the model and evaluates it on a given dataset.

## Customization

- Adjust feature engineering or model parameters in `train_model.py` as needed.
- To use a different threshold for classification, modify the commented-out threshold logic in `test_model.py`.

## Visualization

A word cloud of the combined text fields is generated during training for exploratory analysis.

## License

This project is for educational and research purposes.

---

## Video Link

Please find the video link here - https://drive.google.com/file/d/1IF6ihrJo83aumuNnp0-CaLrKJH5m619Q/view?usp=sharing
