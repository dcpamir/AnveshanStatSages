import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
import joblib
import train_model

# Load model and evaluate F1 score
def load_and_evaluate(file_path, model_path='rf_model_with_transformers.joblib'):
    # Load the pre-trained model and transformers
    rf_model, tfidf, scaler, encoder = joblib.load(model_path)
    print(f"Model and transformers loaded from {model_path}")
    
    features, y = train_model.test_model(file_path)
    
    # Predict on the full dataset
    y_pred = rf_model.predict(features)
    
    f1 = f1_score(y, y_pred)
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y, y_pred))

def load_and_evaluate(file_path, model_path='rf_model_with_transformers.joblib'):

    rf_model, tfidf, scaler, encoder = joblib.load(model_path)
    print(f"Model and transformers loaded from {model_path}")
    
    features, y , df = train_model.test_model(file_path)

    y_pred = rf_model.predict(features)
    y_proba = rf_model.predict_proba(features)[:, 1]  # For histogram and top-10
    
    # Calculate metrics
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    # Table of results
    results = pd.DataFrame({
        'Metric': ['F1 Score', 'Precision', 'Recall'],
        'Class 0 (Real)': [precision_score(y, y_pred, pos_label=0), 
                          precision_score(y, y_pred, pos_label=0), 
                          recall_score(y, y_pred, pos_label=0)],
        'Class 1 (Fake)': [precision, recall, f1],
        'Macro Avg': [precision_score(y, y_pred, average='macro'),
                     recall_score(y, y_pred, average='macro'),
                     f1_score(y, y_pred, average='macro')]
    })
    print("\nTable of Results:")
    print(results.to_string(index=False))
    
    # Histogram of fraud probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Fraud Probabilities')
    plt.xlabel('Probability of Fraud')
    plt.ylabel('Frequency')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
    plt.legend()
    plt.show()
    
    # Pie chart of fake vs. real jobs
    y_pred_labels = ['Real' if pred == 0 else 'Fake' for pred in y_pred]
    pie_data = pd.Series(y_pred_labels).value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['lightgreen', 'salmon'])
    plt.title('Proportion of Predicted Real vs. Fake Jobs')
    plt.show()
    
    # Top-10 most suspicious listings (using index as identifier)
    prob_df = pd.DataFrame({
        'index': df.index,
        'fraud_probability': y_proba
    })
    top_10_suspicious = prob_df.sort_values(by='fraud_probability', ascending=False).head(10)
    print("\nTop-10 Most Suspicious Listings (by Index):")
    print(top_10_suspicious.to_string(index=False))
    
    # Classification report for completeness
    print("\nClassification Report (Default Threshold 0.5):")
    print(classification_report(y, y_pred))

def main(file_path):
    load_and_evaluate(file_path)

if __name__ == '__main__':
    main('job_postings.csv')