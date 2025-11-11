import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(file_path=None):
    """
    Carga un dataset. Si file_path se proporciona, carga el CSV.
    Si no, genera un dataset sintético para pruebas.
    """
    if file_path:
        df = pd.read_csv(file_path)
    else:
        # Generar dataset sintético para pruebas rápidas
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=10000, 
            n_features=16,
            n_informative=12,
            n_redundant=4,
            n_classes=7,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(16)])
        df['Class'] = y
        
        print(f"Dataset sintético generado: {df.shape}")
    
    return df

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """
    Preprocesa los datos: separa features y target, divide en train/test
    """
    # Separar features y target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Codificar etiquetas si son categóricas
    le = None
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Estandarizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': le
    }