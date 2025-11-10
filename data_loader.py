import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(file_path=None, dataset_name='wine'):
    """
    Carga un dataset. Si no se proporciona file_path, usa un dataset de sklearn.
    
    Dataset sugerido: Wine Quality (UCI) o cualquier dataset con >10k filas
    """
    if file_path:
        df = pd.read_csv(file_path)
    else:
        # Usar un dataset más pequeño para pruebas rápidas
        from sklearn.datasets import load_wine, load_iris
        from sklearn.datasets import make_classification
        
        # Opción 1: Generar dataset sintético con 10k filas
        X, y = make_classification(
            n_samples=10000, 
            n_features=7, 
            n_informative=5,
            n_redundant=2,
            n_classes=5,  # Más de 2 clases
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(7)])
        df['target'] = y
        
        print(f"Dataset sintético generado: {df.shape}")
    
    return df

def expand_dataset(df, target_rows=10000):
    """Expande el dataset agregando variaciones con ruido"""
    current_rows = len(df)
    if current_rows >= target_rows:
        return df
    
    multiplier = (target_rows // current_rows) + 1
    expanded_df = pd.concat([df] * multiplier, ignore_index=True)
    
    # Agregar ruido pequeño a las columnas numéricas
    numeric_cols = expanded_df.select_dtypes(include=[np.number]).columns
    if 'target' in numeric_cols:
        numeric_cols = numeric_cols.drop('target')
    
    for col in numeric_cols:
        noise = np.random.normal(0, 0.01 * expanded_df[col].std(), len(expanded_df))
        expanded_df[col] = expanded_df[col] + noise
    
    return expanded_df.head(target_rows)

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """
    Preprocesa los datos: separa features y target, divide en train/test
    """
    # Separar features y target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Codificar etiquetas si son categóricas
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
        'label_encoder': le if 'le' in locals() else None
    }

if __name__ == "__main__":
    # Ejemplo de uso
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target distribution:\n{df['target'].value_counts()}")