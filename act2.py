import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG_FILE = 'config.json'

def find_drybean_dataset():
    """Busca el archivo DryBeanDataset en el directorio actual"""
    possible_names = [
        'DryBeanDataset.csv',
    ]
    
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    # Si no encuentra ninguno, buscar cualquier CSV que contenga "DryBean" o "Bean"
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for csv_file in csv_files:
        if 'bean' in csv_file.lower() or 'dry' in csv_file.lower():
            return csv_file
    
    raise FileNotFoundError("No se encontró el archivo DryBeanDataset.csv en el directorio actual")

def iterate_minibatches(X, y, batch_size, shuffle=True):
    """Itera sobre minibatches de los datos"""
    assert len(X) == len(y)
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
        
    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))
        excerpt = indices[start_idx:end_idx]
        yield X[excerpt], y[excerpt]

def load_data(data_file, test_size):
    """Lee los datos del CSV, procesa las clases y divide los datos de entrenamiento y testing"""
    data = pd.read_csv(data_file, header=0)
    
    X = data.iloc[:, :-1]
    y_raw = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    all_classes = np.unique(y)
    print(f"Clases detectadas: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, all_classes

def initialize_models(config):
    """Inicializa modelos según la configuración"""
    model_tracker = []
    
    for model_config in config['model_configs']:
        model_name = model_config['name']
        model_params = model_config['params'].copy()
        
        batch_size = model_params.pop('batch_size')
        
        model = SGDClassifier(
            **model_params,
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        
        model_tracker.append({
            "name": model_name,
            "model": model,
            "batch_size": batch_size,
            "params": model_config['params'], 
            "active": True,
            "train_score": 0.0
        })
    
    print(f"Inicializados {len(model_tracker)} modelos.")
    return model_tracker

def main():
    # Buscar archivo DryBeanDataset
    data_file = find_drybean_dataset()
    print(f"Archivo encontrado: {data_file}\n")
    
    # Cargar configuración
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    settings = config['training_settings']
    csv_data = load_data(data_file, settings['test_split_size'])
    
    X_train, X_test, y_train, y_test, label_encoder, all_classes = csv_data
    model_tracker = initialize_models(config)
    active_models = model_tracker.copy()
    
    print("\n Ciclo de entrenamiento por epocas: ")
    
    # dentro de este for empieza el ciclo de entrenamiento y descarte de modelos
    for epoch in range(1, settings['total_epochs'] + 1):
        print(f"\n Epoca {epoch}/{settings['total_epochs']}")
        
        # entrenamiento por epoca
        for config_dict in active_models:
            model = config_dict['model']
            batch_size = config_dict['batch_size']
            
            for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
                model.partial_fit(X_batch, y_batch, classes=all_classes)
        
        # dentro de este if esta la logica de descarte
        if epoch % settings['pruning_interval_epochs'] == 0:
            print(f"\nEpoca {epoch}: Ciclo de evaluacion de modelos y descartes")
            
            if len(active_models) <= settings['num_final_models']:
                print("Quedan solo los modelos finales, no se descarta mas")
                continue
            
            # calcula la precision de cada algoritmo para evaluar cuales pasan y cuales se descartan
            for config_dict in active_models:
                score = config_dict['model'].score(X_train, y_train)
                config_dict['train_score'] = score
                print(f"  Modelo: {config_dict['name']:<20}, Acc: {score:.4f}")
            
            active_models.sort(key=lambda x: x['train_score'])
            worst_model_config = active_models.pop(0) 
            worst_model_config['active'] = False
            
            print(f"Se decarta: {worst_model_config['name']}, con Acc = {worst_model_config['train_score']:.4f}")
            print(f"Quedan {len(active_models)} modelos activos")
    
    print(f"\n\nEntrenamiento completado")
    print(f"{settings['num_final_models']} mejores modelos en el conjunto de PRUEBA (20%).")    
    # evalua los modelos que quedaron en el arreglo 
    final_best_models = active_models
    
    for config_dict in final_best_models:
        model = config_dict['model']
        name = config_dict['name']
        
        print(f"\nModelo Final: {name}")
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy en Test: {acc:.4f}\n")
        
        print("Reporte de Clasificacion (Test):")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=label_encoder.classes_
        ))
        print("Hiperparametros usados:")
        print(json.dumps(config_dict['params'], indent=2))
        print("---------------------------------------------------")

if __name__ == "__main__":
    main()
