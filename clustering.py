import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
import pandas as pd
from collections import Counter

class ClusteringComparison:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
        
    def run_kmeans(self, n_clusters_list=[3, 5, 7, 9], random_state=42):
        """Ejecuta K-Means con diferentes configuraciones"""
        results = []
        for n_clusters in n_clusters_list:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, 
                          n_init=10, max_iter=300)
            labels = kmeans.fit_predict(self.X_train)
            score = silhouette_score(self.X_train, labels)
            
            results.append({
                'algorithm': 'K-Means',
                'config': f'n_clusters={n_clusters}',
                'n_clusters': n_clusters,
                'model': kmeans,
                'score': score,
                'labels_train': labels
            })
        return results
    
    def run_kmeans_plus_plus(self, n_clusters_list=[3, 5, 7, 9], random_state=42):
        """Ejecuta K-Means++ con diferentes configuraciones"""
        results = []
        for n_clusters in n_clusters_list:
            # K-Means++ es la inicialización por defecto en sklearn
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', 
                          random_state=random_state, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(self.X_train)
            score = silhouette_score(self.X_train, labels)
            
            results.append({
                'algorithm': 'K-Means++',
                'config': f'n_clusters={n_clusters} (k-means++ init)',
                'n_clusters': n_clusters,
                'model': kmeans,
                'score': score,
                'labels_train': labels
            })
        return results
    
    def run_meanshift(self, bandwidth_list=None, quantile_list=[0.2, 0.3, 0.4, 0.5], 
                     random_state=42):
        """Ejecuta MeanShift con diferentes configuraciones"""
        if bandwidth_list is None:
            # Estimar bandwidth automáticamente usando diferentes quantiles
            bandwidth_list = []
            for quantile in quantile_list:
                bw = estimate_bandwidth(self.X_train, quantile=quantile, 
                                      n_samples=min(1000, len(self.X_train)))
                bandwidth_list.append(bw)
        
        results = []
        for bandwidth in bandwidth_list:
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, 
                                min_bin_freq=1, n_jobs=-1)
            labels = meanshift.fit_predict(self.X_train)
            n_clusters = len(np.unique(labels))
            
            if n_clusters > 1:  # Silhouette score requiere al menos 2 clusters
                score = silhouette_score(self.X_train, labels)
            else:
                score = -1  # Penalizar si solo hay 1 cluster
            
            results.append({
                'algorithm': 'MeanShift',
                'config': f'bandwidth={bandwidth:.3f}',
                'bandwidth': bandwidth,
                'n_clusters': n_clusters,
                'model': meanshift,
                'score': score,
                'labels_train': labels
            })
        return results
    
    def evaluate_all_configurations(self):
        """Ejecuta todas las configuraciones y las evalúa"""
        print("Ejecutando K-Means...")
        kmeans_results = self.run_kmeans()
        
        print("Ejecutando K-Means++...")
        kmeans_pp_results = self.run_kmeans_plus_plus()
        
        print("Ejecutando MeanShift...")
        meanshift_results = self.run_meanshift()
        
        # Combinar todos los resultados
        all_results = kmeans_results + kmeans_pp_results + meanshift_results
        self.results = all_results
        
        # Ordenar por score
        all_results_sorted = sorted(all_results, key=lambda x: x['score'], reverse=True)
        
        print("\n=== Resultados de todas las configuraciones ===")
        for i, result in enumerate(all_results_sorted, 1):
            print(f"{i}. {result['algorithm']} - {result['config']}: "
                  f"Silhouette Score = {result['score']:.4f}")
        
        return all_results_sorted
    
    def select_top_configurations(self, top_n=3):
        """Selecciona las top N configuraciones"""
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        top_results = sorted_results[:top_n]
        
        print(f"\n=== Top {top_n} configuraciones seleccionadas ===")
        for i, result in enumerate(top_results, 1):
            print(f"{i}. {result['algorithm']} - {result['config']}: "
                  f"Score = {result['score']:.4f}")
        
        return top_results
    
    def apply_to_test_set(self, model, algorithm_name):
        """Aplica el modelo entrenado al conjunto de test"""
        labels_test = model.predict(self.X_test)
        return labels_test
    
    def analyze_cluster_labels(self, labels_test, algorithm_name, config_name):
        """Analiza la correspondencia entre clusters y etiquetas reales"""
        # Para cada cluster, encontrar la etiqueta dominante
        cluster_label_map = {}
        cluster_counts = {}
        
        # Analizar en el conjunto de entrenamiento primero
        # (Asumiendo que tenemos acceso a y_train a través de self)
        # Pero como no usamos y_train en clustering, necesitamos una estrategia diferente
        
        # Estrategia: Usar el conjunto de test para mapear clusters a etiquetas
        for cluster_id in np.unique(labels_test):
            mask = labels_test == cluster_id
            cluster_labels = self.y_test[mask]
            
            if len(cluster_labels) > 0:
                # Encontrar la etiqueta más común en este cluster
                label_counts = Counter(cluster_labels)
                dominant_label = label_counts.most_common(1)[0][0]
                cluster_label_map[cluster_id] = dominant_label
                cluster_counts[cluster_id] = len(cluster_labels)
        
        # Predecir etiquetas basadas en clusters
        predicted_labels = np.array([cluster_label_map.get(c, -1) for c in labels_test])
        
        # Calcular precisión
        accuracy = np.mean(predicted_labels == self.y_test)
        
        # Calcular distribución de etiquetas por cluster
        cluster_analysis = {}
        for cluster_id in np.unique(labels_test):
            mask = labels_test == cluster_id
            cluster_labels = self.y_test[mask]
            label_dist = Counter(cluster_labels)
            cluster_analysis[cluster_id] = {
                'total_samples': len(cluster_labels),
                'label_distribution': dict(label_dist),
                'dominant_label': cluster_label_map.get(cluster_id, None)
            }
        
        return {
            'predicted_labels': predicted_labels,
            'accuracy': accuracy,
            'cluster_label_map': cluster_label_map,
            'cluster_analysis': cluster_analysis
        }
    
    def evaluate_top_configurations_on_test(self, top_configurations):
        """Evalúa las mejores configuraciones en el conjunto de test"""
        test_results = []
        
        for config in top_configurations:
            print(f"\nEvaluando {config['algorithm']} - {config['config']} en test set...")
            
            # Aplicar al conjunto de test
            labels_test = self.apply_to_test_set(config['model'], config['algorithm'])
            
            # Analizar correspondencia con etiquetas reales
            analysis = self.analyze_cluster_labels(
                labels_test, config['algorithm'], config['config']
            )
            
            # Calcular silhouette score en test set
            test_silhouette = silhouette_score(self.X_test, labels_test)
            
            result = {
                'algorithm': config['algorithm'],
                'config': config['config'],
                'test_silhouette': test_silhouette,
                'label_accuracy': analysis['accuracy'],
                'cluster_analysis': analysis['cluster_analysis'],
                'n_clusters': len(np.unique(labels_test))
            }
            
            test_results.append(result)
            
            print(f"  Test Silhouette Score: {test_silhouette:.4f}")
            print(f"  Label Assignment Accuracy: {analysis['accuracy']:.4f}")
            print(f"  Number of clusters: {len(np.unique(labels_test))}")
        
        return test_results

def justify_parameters():
    """Justifica la elección de parámetros"""
    justification = {
        'K-Means': {
            'n_clusters': [3, 5, 7, 9],
            'reason': 'Variar el número de clusters permite explorar diferentes niveles de granularidad. Valores impares ayudan a evitar empates en la asignación.'
        },
        'K-Means++': {
            'n_clusters': [3, 5, 7, 9],
            'reason': 'Mismos valores que K-Means para comparación justa. K-Means++ mejora la inicialización reduciendo la dependencia de la inicialización aleatoria.'
        },
        'MeanShift': {
            'bandwidth': 'Quantiles [0.2, 0.3, 0.4, 0.5]',
            'reason': 'El bandwidth determina el tamaño de la ventana de búsqueda. Quantiles menores producen más clusters (más granular), mayores producen menos clusters (más general).'
        }
    }
    return justification