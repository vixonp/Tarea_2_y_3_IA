import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification

def load_dataset(file_path=None):
    if file_path:
        df = pd.read_csv(file_path)
    else:
        X, y = make_classification(
            n_samples=10000, n_features=16, n_informative=12, n_redundant=4,
            n_classes=7, random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(16)])
        df['Class'] = y
    return df

def preprocess_data(df, target_column='Class', test_size=0.2, random_state=42):
    X, y = df.drop(columns=[target_column]), df[target_column]
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    return {
        'X_train': scaler.fit_transform(X_train),
        'X_test': scaler.transform(X_test),
        'y_train': y_train,
        'y_test': y_test
    }
