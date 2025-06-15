import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def generate_word_cloud(df):
  # Combine text from relevant columns
  text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
  text = " ".join(df[col].astype(str).str.cat(sep=' ') for col in text_columns)

  # Generate a word cloud image
  wordcloud = WordCloud(random_state=42, width = 800, height = 450).generate(text)

  # Display the generated image:
  # the matplotlib way:
  plt.figure(figsize=(10, 8))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()

# Load and preprocess data
def load_and_clean_data(df):

    df.drop(['job_id'],axis=1,inplace = True)
    X = df.drop(columns=['fraudulent'])
    y = df['fraudulent']

    # Initialize NLP tools
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    # Clean text features
    def clean_text(text):
        if isinstance(text, str):
            words = word_tokenize(text.lower())
            words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
            return ' '.join(words)
        return ''

    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        X[col] = X[col].apply(clean_text)

    # Create text_specified binary feature
    X['text_specified'] = X[text_cols].apply(lambda x: 1 if any(x.dropna().astype(str).str.strip() != '') else 0, axis=1)

    # Decompose location
    def split_location(loc):
        if isinstance(loc, str) and ',' in loc:
            parts = [part.strip() for part in loc.split(',')]
            return [parts[0] if len(parts) > 0 else 'Unspecified',
                    parts[1] if len(parts) > 1 else 'Unspecified',
                    parts[2] if len(parts) > 2 else 'Unspecified']
        return ['Unspecified', 'Unspecified', 'Unspecified']

    loc_data = X['location'].apply(split_location).apply(pd.Series)
    X['country'] = loc_data[0]
    X['state'] = loc_data[1]
    X['city'] = loc_data[2]
    X = X.drop(columns=['location'])

    # Decompose salary_range
    def split_salary(salary):
        if isinstance(salary, str) and '-' in salary:
            try:
                min_sal, max_sal = map(int, salary.split('-'))
                return [min_sal, max_sal, 1]
            except:
                return [0, 0, 0]
        return [0, 0, 0]

    sal_data = X['salary_range'].apply(split_salary).apply(pd.Series)
    X['min_salary'] = sal_data[0]
    X['max_salary'] = sal_data[1]
    X['salary_specified'] = sal_data[2]
    X = X.drop(columns=['salary_range'])

    # Handle missing categorical values
    cat_cols = ['department', 'employment_type', 'required_experience', 'required_education',
                'industry', 'function', 'country', 'state', 'city']
    for col in cat_cols:
        X[col] = X[col].fillna('Unspecified')

    return X, y

# Feature engineering
def engineer_features(X):
    # TF-IDF for text features
    # Text features
    tfidf = TfidfVectorizer(max_features=500)
    text_cols = ['title', 'company_profile', 'description',
                'requirements', 'benefits']
    text_features = tfidf.fit_transform(X[text_cols].agg(' '.join, axis=1)).toarray()

    # Numerical and binary features (exclude job_id)
    num_cols = ['min_salary', 'max_salary', 'telecommuting', 'has_company_logo',
                'has_questions', 'salary_specified']
    scaler = StandardScaler()
    num_features = scaler.fit_transform(X[num_cols])

    cat_cols = ['country', 'state', 'city']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_features = encoder.fit_transform(X[cat_cols])

    # Combine features
    features = np.hstack([text_features, cat_features, num_features])
    return features, tfidf, scaler, encoder

def train_and_evaluate(X, y):

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, 
                                                    stratify=y_sm, random_state=42)
    rf = RandomForestClassifier(class_weight='balanced', max_depth=None, min_samples_split=2, n_estimators=200)
    rf.fit(X_train, y_train)
    y_pred_class = rf.predict(X_test)

    print("Classification Accuracy:", accuracy_score(y_test, y_pred_class))
    print("Classification Report\n")
    print(classification_report(y_test, y_pred_class))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_class))

    return rf

def save_model(rf, tfidf, scaler, encoder):
    joblib.dump((rf, tfidf, scaler, encoder), 'rf_model_with_transformers.joblib')

def test_model(file_path):

    df = pd.read_csv(file_path)

    generate_word_cloud(df)

    X, y = load_and_clean_data(df)

    features, tfidf, scaler, encoder  = engineer_features(X)

    return features, y, df

def main(file_path):

    df = pd.read_csv(file_path)

    #Visualisation
    generate_word_cloud(df)

    # Load and preprocess
    X, y = load_and_clean_data(df)

    # Feature engineering
    features, tfidf, scaler, encoder  = engineer_features(X)

    # Train and evaluate
    rf = train_and_evaluate(features, y)

    save_model(rf, tfidf, scaler, encoder)

    print("MLP training and evaluation complete.")

if __name__ == '__main__':
    main('job_postings.csv')