# data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps
    return processed_data

def split_data(data, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data['padded_sequences'], data['labels'], test_size=test_size)
    return X_train, X_test, y_train, y_test
