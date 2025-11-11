import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from collections import Counter
import time

class ClusteringComparison:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
    
    def _get_kmeans_params(self):
        """Determina parámetros óptimos según tamaño del dataset"""
        n_samples = len(self.X_train)
        if n_samples > 50000:
            return {'n_init': 3, 'max_iter': 100}
        return {'n_init': 10, 'max_iter': 300}
    
    def _calculate_silhouette(self, data, labels):
        """Calcula silhouette score, usando muestra si el dataset es grande"""
        n_samples = len(data)
        if n_samples > 10000:
            sample_size = min(5000, n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            score = silhouette_score(data[indices], labels[indices])
            return score, f"(muestra de {sample_size})"
        return silhouette_score(data, labels), ""
        
    def run_kmeans(self, n_clusters_list=[3, 5, 7, 9], init='k-means++', random_state=42):
        """Ejecuta K-Means con diferentes configuraciones"""
        results = []
        algo_name = 'K-Means++' if init == 'k-means++' else 'K-Means'
        params = self._get_kmeans_params()
        
        for idx, n_clusters in enumerate(n_clusters_list, 1):
            print(f"  {algo_name}: Config {idx}/{len(n_clusters_list)} (n_clusters={n_clusters})...", end=' ', flush=True)
            start_time = time.time()
            
            kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=random_state, **params)
            labels = kmeans.fit_predict(self.X_train)
            score, sample_info = self._calculate_silhouette(self.X_train, labels)
            
            elapsed = time.time() - start_time
            print(f"Silhouette {sample_info}: {score:.4f} ({elapsed:.1f}s)")
            
            results.append({
                'algorithm': algo_name,
                'config': f'n_clusters={n_clusters}' + ('' if init == 'k-means++' else ' (random init)'),
                'n_clusters': n_clusters,
                'model': kmeans,
                'score': score,
                'labels_train': labels
            })
        return results
    
    def run_meanshift(self, bandwidth_list=None, quantile_list=[0.2, 0.3, 0.4, 0.5], random_state=42):
        """Ejecuta MeanShift con diferentes configuraciones"""
        if bandwidth_list is None:
            bandwidth_list = []
            print("  MeanShift: Estimando bandwidths...", end=' ', flush=True)
            n_samples = len(self.X_train)
            sample_size = min(5000, n_samples)
            sample_data = self.X_train[np.random.choice(n_samples, sample_size, replace=False)]
            
            for quantile in quantile_list:
                bw = estimate_bandwidth(sample_data, quantile=quantile, n_samples=min(1000, sample_size))
                bandwidth_list.append(bw)
            print("✓")
        
        results = []
        for idx, bandwidth in enumerate(bandwidth_list, 1):
            print(f"  MeanShift: Config {idx}/{len(bandwidth_list)} (bandwidth={bandwidth:.3f})...", end=' ', flush=True)
            start_time = time.time()
            
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=1, n_jobs=-1)
            labels = meanshift.fit_predict(self.X_train)
            n_clusters = len(np.unique(labels))
            
            if n_clusters > 1:
                score, sample_info = self._calculate_silhouette(self.X_train, labels)
                print(f"Silhouette {sample_info}: {score:.4f}, Clusters: {n_clusters}", end='')
            else:
                score = -1
                print(f"Silhouette: {score:.4f} (1 cluster)", end='')
            
            elapsed = time.time() - start_time
            print(f" ({elapsed:.1f}s)")
            
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
        print("\nEjecutando K-Means...")
        kmeans_results = self.run_kmeans(init='random')
        
        print("\nEjecutando K-Means++...")
        kmeans_pp_results = self.run_kmeans(init='k-means++')
        
        print("\nEjecutando MeanShift...")
        meanshift_results = self.run_meanshift()
        
        self.results = kmeans_results + kmeans_pp_results + meanshift_results
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        print("\n=== Resultados de todas las configuraciones ===")
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['algorithm']} - {result['config']}: Silhouette = {result['score']:.4f}")
        
        return sorted_results
    
    def select_top_configurations(self, top_n=3):
        """Selecciona las top N configuraciones"""
        top_results = sorted(self.results, key=lambda x: x['score'], reverse=True)[:top_n]
        
        print(f"\n=== Top {top_n} configuraciones seleccionadas ===")
        for i, result in enumerate(top_results, 1):
            print(f"{i}. {result['algorithm']} - {result['config']}: Score = {result['score']:.4f}")
        
        return top_results
    
    def analyze_cluster_labels(self, labels_test):
        """Analiza la correspondencia entre clusters y etiquetas reales"""
        cluster_label_map = {}
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels_test):
            mask = labels_test == cluster_id
            cluster_labels = self.y_test[mask]
            
            if len(cluster_labels) > 0:
                label_counts = Counter(cluster_labels)
                dominant_label = label_counts.most_common(1)[0][0]
                cluster_label_map[cluster_id] = dominant_label
                
                cluster_analysis[cluster_id] = {
                    'total_samples': len(cluster_labels),
                    'label_distribution': dict(label_counts),
                    'dominant_label': dominant_label
                }
        
        predicted_labels = np.array([cluster_label_map.get(c, -1) for c in labels_test])
        accuracy = np.mean(predicted_labels == self.y_test)
        
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
            
            labels_test = config['model'].predict(self.X_test)
            analysis = self.analyze_cluster_labels(labels_test)
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
    return {
        'K-Means/K-Means++': {
            'n_clusters': [3, 5, 7, 9],
            'reason': 'Variar el número de clusters permite explorar diferentes niveles de granularidad. K-Means++ mejora la inicialización.'
        },
        'MeanShift': {
            'bandwidth': 'Quantiles [0.2, 0.3, 0.4, 0.5]',
            'reason': 'El bandwidth determina el tamaño de la ventana. Quantiles menores → más clusters, mayores → menos clusters.'
        }
    }