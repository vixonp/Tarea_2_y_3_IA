import json
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from copy import deepcopy

class ParallelModelTraining:
    def __init__(self, X_train, X_test, y_train, y_test, config_path='config.json'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.config_path = config_path
        self.config = self.load_config()
        self.models = []
        self.training_history = []
        self.lock = threading.Lock()
        
    def load_config(self):
        """Carga la configuración desde archivo JSON"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def create_logistic_regression_model(self, config):
        """Crea un modelo de Regresión Logística según la configuración"""
        # Usar SGDClassifier para controlar learning_rate y batch_size
        if 'solver' in config and config['solver'] == 'lbfgs':
            # Para LBFGS, usar LogisticRegression estándar
            model = LogisticRegression(
                C=config.get('C', 1.0),
                max_iter=config.get('max_iter', 100),
                random_state=42,
                n_jobs=1
            )
        else:
            # Para SGD, usar SGDClassifier
            model = SGDClassifier(
                loss='log_loss',
                learning_rate='constant',
                eta0=config.get('learning_rate', 0.01),
                max_iter=config.get('max_iter', 100),
                random_state=42,
                warm_start=True,
                alpha=1.0 / config.get('C', 1.0)
            )
        
        return {
            'name': config['name'],
            'type': 'LogisticRegression',
            'config': config,
            'model': model,
            'epoch': 0,
            'scores': []
        }
    
    def create_svm_model(self, config):
        """Crea un modelo SVM según la configuración"""
        model = SVC(
            C=config.get('C', 1.0),
            kernel=config.get('kernel', 'rbf'),
            gamma=config.get('gamma', 'scale'),
            degree=config.get('degree', 3),
            max_iter=config.get('max_iter', 100),
            random_state=42
        )
        
        return {
            'name': config['name'],
            'type': 'SVM',
            'config': config,
            'model': model,
            'epoch': 0,
            'scores': []
        }
    
    def initialize_models(self):
        """Inicializa todos los modelos según la configuración"""
        self.models = []
        
        # Crear modelos de Regresión Logística
        for lr_config in self.config['logistic_regression']:
            model_wrapper = self.create_logistic_regression_model(lr_config)
            self.models.append(model_wrapper)
        
        # Crear modelos SVM
        for svm_config in self.config['svm']:
            model_wrapper = self.create_svm_model(svm_config)
            self.models.append(model_wrapper)
        
        print(f"Inicializados {len(self.models)} modelos:")
        for m in self.models:
            print(f"  - {m['name']} ({m['type']})")
    
    def train_model_epoch(self, model_wrapper, batch_size=None):
        """Entrena un modelo por una época"""
        model = model_wrapper['model']
        config = model_wrapper['config']
        
        # Para modelos que soportan partial_fit (SGDClassifier)
        if hasattr(model, 'partial_fit'):
            if batch_size is None:
                batch_size = config.get('batch_size', 32)
            
            # Dividir datos en batches
            n_samples = len(self.X_train)
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = self.X_train[batch_indices]
                y_batch = self.y_train[batch_indices]
                
                # Si es la primera vez, necesita ver todas las clases
                if model_wrapper['epoch'] == 0 and i == 0:
                    model.partial_fit(X_batch, y_batch, classes=np.unique(self.y_train))
                else:
                    model.partial_fit(X_batch, y_batch)
        else:
            # Para modelos que no soportan partial_fit, reentrenar completamente
            # Esto es menos eficiente pero necesario para algunos modelos
            model.fit(self.X_train, self.y_train)
        
        model_wrapper['epoch'] += 1
        
        # Evaluar en conjunto de entrenamiento
        y_pred = model.predict(self.X_train)
        accuracy = accuracy_score(self.y_train, y_pred)
        f1 = f1_score(self.y_train, y_pred, average='weighted')
        
        return accuracy, f1
    
    def evaluate_model(self, model_wrapper):
        """Evalúa un modelo en el conjunto de entrenamiento"""
        model = model_wrapper['model']
        y_pred = model.predict(self.X_train)
        accuracy = accuracy_score(self.y_train, y_pred)
        f1 = f1_score(self.y_train, y_pred, average='weighted')
        return accuracy, f1
    
    def train_all_models_parallel(self, epochs=100, eval_interval=5):
        """Entrena todos los modelos en paralelo con evaluación periódica"""
        self.initialize_models()
        
        current_epoch = 0
        active_models = self.models.copy()
        
        print(f"\nIniciando entrenamiento paralelo por {epochs} épocas...")
        print(f"Evaluación cada {eval_interval} épocas. Descarte del peor modelo cada evaluación.\n")
        
        while current_epoch < epochs and len(active_models) > 1:
            # Entrenar todos los modelos activos en paralelo
            with ThreadPoolExecutor(max_workers=len(active_models)) as executor:
                futures = []
                for model_wrapper in active_models:
                    future = executor.submit(self.train_model_epoch, model_wrapper)
                    futures.append((future, model_wrapper))
                
                # Esperar a que todos terminen
                for future, model_wrapper in futures:
                    try:
                        acc, f1 = future.result()
                        with self.lock:
                            model_wrapper['scores'].append({
                                'epoch': model_wrapper['epoch'],
                                'accuracy': acc,
                                'f1_score': f1
                            })
                    except Exception as e:
                        print(f"Error entrenando {model_wrapper['name']}: {e}")
            
            current_epoch += 1
            
            # Evaluar periódicamente
            if current_epoch % eval_interval == 0:
                print(f"\n=== Época {current_epoch} - Evaluación ===")
                
                # Evaluar todos los modelos activos
                model_scores = []
                for model_wrapper in active_models:
                    acc, f1 = self.evaluate_model(model_wrapper)
                    model_scores.append({
                        'model': model_wrapper,
                        'accuracy': acc,
                        'f1_score': f1
                    })
                    print(f"  {model_wrapper['name']}: Accuracy={acc:.4f}, F1={f1:.4f}")
                
                # Ordenar por desempeño (usando F1 como métrica principal)
                model_scores.sort(key=lambda x: x['f1_score'], reverse=True)
                
                # Descarte del peor modelo (si hay más de uno)
                if len(model_scores) > 1:
                    worst_model = model_scores[-1]['model']
                    active_models.remove(worst_model)
                    print(f"\n  ❌ Descartado: {worst_model['name']} (F1={model_scores[-1]['f1_score']:.4f})")
                    print(f"  ✅ Modelos activos restantes: {len(active_models)}")
        
        print(f"\n=== Entrenamiento completado ===")
        print(f"Modelos finales: {len(active_models)}")
        for model_wrapper in active_models:
            print(f"  - {model_wrapper['name']} ({model_wrapper['type']})")
        
        self.models = active_models
        return active_models
    
    def select_best_models(self, top_n=2):
        """Selecciona los mejores modelos (uno de cada tipo si es posible)"""
        # Separar por tipo
        lr_models = [m for m in self.models if m['type'] == 'LogisticRegression']
        svm_models = [m for m in self.models if m['type'] == 'SVM']
        
        best_models = []
        
        # Seleccionar el mejor de cada tipo
        if lr_models:
            best_lr = max(lr_models, key=lambda x: x['scores'][-1]['f1_score'] if x['scores'] else 0)
            best_models.append(best_lr)
        
        if svm_models:
            best_svm = max(svm_models, key=lambda x: x['scores'][-1]['f1_score'] if x['scores'] else 0)
            best_models.append(best_svm)
        
        # Si necesitamos más modelos, tomar los mejores en general
        if len(best_models) < top_n:
            all_models = sorted(self.models, 
                              key=lambda x: x['scores'][-1]['f1_score'] if x['scores'] else 0,
                              reverse=True)
            for model in all_models:
                if model not in best_models and len(best_models) < top_n:
                    best_models.append(model)
        
        print(f"\n=== Mejores {len(best_models)} modelos seleccionados ===")
        for model in best_models:
            if model['scores']:
                final_score = model['scores'][-1]
                print(f"  - {model['name']} ({model['type']}): "
                      f"Accuracy={final_score['accuracy']:.4f}, "
                      f"F1={final_score['f1_score']:.4f}")
        
        return best_models
    
    def evaluate_on_test_set(self, models):
        """Evalúa los mejores modelos en el conjunto de test"""
        test_results = []
        
        print(f"\n=== Evaluación en conjunto de test ===")
        for model_wrapper in models:
            model = model_wrapper['model']
            
            # Predecir en test set
            y_pred = model.predict(self.X_test)
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_micro = f1_score(self.y_test, y_pred, average='micro')
            
            # Reporte detallado
            report = classification_report(self.y_test, y_pred, output_dict=True)
            
            result = {
                'name': model_wrapper['name'],
                'type': model_wrapper['type'],
                'config': model_wrapper['config'],
                'test_accuracy': accuracy,
                'test_f1_weighted': f1,
                'test_f1_macro': f1_macro,
                'test_f1_micro': f1_micro,
                'classification_report': report
            }
            
            test_results.append(result)
            
            print(f"\n{model_wrapper['name']} ({model_wrapper['type']}):")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 (weighted): {f1:.4f}")
            print(f"  F1 (macro): {f1_macro:.4f}")
            print(f"  F1 (micro): {f1_micro:.4f}")
        
        return test_results
    
    def analyze_hyperparameters(self, test_results):
        """Analiza el impacto de los hiperparámetros en el desempeño"""
        print(f"\n=== Análisis de Hiperparámetros ===")
        
        for result in test_results:
            print(f"\n{result['name']} ({result['type']}):")
            print(f"  Configuración: {result['config']}")
            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"  Test F1 (weighted): {result['test_f1_weighted']:.4f}")
            
            # Análisis específico por tipo
            if result['type'] == 'LogisticRegression':
                config = result['config']
                print(f"  - Learning Rate: {config.get('learning_rate', 'N/A')}")
                print(f"  - Batch Size: {config.get('batch_size', 'N/A')}")
                print(f"  - C (regularization): {config.get('C', 'N/A')}")
                
            elif result['type'] == 'SVM':
                config = result['config']
                print(f"  - C (regularization): {config.get('C', 'N/A')}")
                print(f"  - Kernel: {config.get('kernel', 'N/A')}")
                print(f"  - Gamma: {config.get('gamma', 'N/A')}")