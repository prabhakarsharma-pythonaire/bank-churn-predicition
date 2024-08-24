import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

def encode_with_default(df, col, unknown_label='unknown'):
    le = LabelEncoder()
    df[col] = df[col].fillna(unknown_label)  # Fill NaN with unknown
    unique_classes = list(df[col].unique()) + [unknown_label]  # Add unknown label
    le.fit(unique_classes)
    df[col] = le.transform(df[col])
    return le

def preprocess_data(data, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = encode_with_default(data, col, unknown_label='unknown')
    return data, label_encoders

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return accuracy, f1, precision, recall

def save_model(model, label_encoders, scaler, filepath='model.pkl'):
    with open(filepath, 'wb') as f:
        joblib.dump({'model': model, 'label_encoders': label_encoders, 'scaler': scaler}, f)

def load_model(filepath='model.pkl'):
    with open(filepath, 'rb') as f:
        return joblib.load(f)
