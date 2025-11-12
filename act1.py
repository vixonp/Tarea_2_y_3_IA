import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import time
import os

# cargar y preparar datos (train 80%, test 20%)
def load_data(file_path='DryBeanDataset.csv', target='Class'):
    # cargar el dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_csv(file_path)

    # separar features (X) y target (y)
    X = df.drop(columns=[target])
    y = df[target]
    
    # codificar etiquetas si son texto y guardar nombres originales
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        class_names = label_encoder.classes_
    else:
        y_encoded = y
        class_names = np.array([f'Clase_{i}' for i in np.unique(y)])
    
    # split 80-20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, 
                                                          random_state=42, stratify=y_encoded)
    
    # escalar features (normalización)
    scaler = StandardScaler()
    return {
        'X_train': scaler.fit_transform(X_train),
        'X_test': scaler.transform(X_test),
        'y_train': y_train,
        'y_test': y_test,
        'n_classes': len(np.unique(y_encoded)),
        'class_names': class_names
    }

# definir parámetros: 4 configs por algoritmo
def get_parameters(n_classes):
    return {
        'k_values': [n_classes-1, n_classes, n_classes+1, n_classes+2],  # K-Means
        'quantiles': [0.2, 0.3, 0.4, 0.5]  # MeanShift
    }

class ClusteringAnalysis:
    def __init__(self, data):
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.class_names = data['class_names']  # guardar nombres de clases
        self.results = []
    
    # calcular Silhouette Score (con muestreo si dataset es grande)
    def silhouette(self, X, labels):
        if len(np.unique(labels)) <= 1:  # solo 1 cluster -> score inválido
            return -1.0
        if len(X) > 10000:  # dataset grande -> muestrear para velocidad
            idx = np.random.choice(len(X), 5000, replace=False)
            return silhouette_score(X[idx], labels[idx])
        return silhouette_score(X, labels)
    
    # ejecutar K-Means o K-Means++ con 4 valores de k
    def run_kmeans(self, k_values, init='k-means++'):
        name = 'K-Means++' if init == 'k-means++' else 'K-Means'
        results = []
        for k in k_values:
            model = KMeans(n_clusters=k, init=init, n_init=10, max_iter=300, random_state=42)
            labels = model.fit_predict(self.X_train)
            score = self.silhouette(self.X_train, labels)
            results.append({
                'algorithm': name, 'config': f'k={k}', 'n_clusters': k,
                'model': model, 'score': score
            })
        return results
    
    # ejecutar MeanShift con 4 valores
    def run_meanshift(self, quantiles):
        results = []
        # estimar bandwidth en muestra pequeña 
        sample = self.X_train[np.random.choice(len(self.X_train), 3000, replace=False)]
        for q in quantiles:
            bw = estimate_bandwidth(sample, quantile=q, n_samples=1000)
            model = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
            labels = model.fit_predict(self.X_train)
            score = self.silhouette(self.X_train, labels)
            results.append({
                'algorithm': 'MeanShift', 'config': f'q={q}', 
                'n_clusters': len(np.unique(labels)), 'model': model, 'score': score
            })
        return results
    
    # evaluar las 12 configuraciones en train set
    def evaluate_all(self, params):
        print("\n" + "="*70)
        print("ENTRENAMIENTO: 12 CONFIGURACIONES (80% datos)")
        print("="*70)
        
        # 4 K-Means + 4 K-Means++ + 4 MeanShift = 12 configs
        self.results = (self.run_kmeans(params['k_values'], init='random') +
                       self.run_kmeans(params['k_values'], init='k-means++') +
                       self.run_meanshift(params['quantiles']))
        
        # ordenar por Silhouette Score (mayor = mejor)
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        # mostrar tabla de resultados
        print(f"\n{'#':<3} {'Algoritmo':<12} {'Config':<10} {'Silhouette':>11} {'Clusters':>8}")
        print("-"*70)
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:<3} {r['algorithm']:<12} {r['config']:<10} {r['score']:>11.4f} {r['n_clusters']:>8}")
        
        return sorted_results
    
    # evaluar top 3 en test set y comparar con etiquetas reales
    def evaluate_test(self, top3):
        print("\n" + "="*70)
        print("EVALUACIÓN TEST: TOP 3 CONFIGURACIONES (20% datos)")
        print("="*70)
        
        results = []
        for i, cfg in enumerate(top3, 1):
            # predecir clusters en test
            labels_test = cfg['model'].predict(self.X_test)
            test_sil = self.silhouette(self.X_test, labels_test)
            
            # asignar etiqueta dominante a cada cluster
            cluster_map = {}
            analysis = {}
            for c in np.unique(labels_test):
                mask = labels_test == c
                real = self.y_test[mask]
                if len(real) > 0:
                    counts = Counter(real)
                    dominant = counts.most_common(1)[0][0]
                    cluster_map[c] = dominant
                    
                    # convertir índice a nombre de clase
                    dominant_name = self.class_names[dominant]
                    
                    analysis[c] = {
                        'n': len(real),
                        'dominant': dominant,
                        'dominant_name': dominant_name,  # agregar nombre
                        'purity': counts[dominant] / len(real)  # pureza del cluster
                    }
            
            # comparar predicción con etiqueta real
            y_pred = np.array([cluster_map.get(c, -1) for c in labels_test])
            accuracy = np.mean(y_pred == self.y_test)
            
            results.append({
                'algorithm': cfg['algorithm'], 'config': cfg['config'],
                'test_sil': test_sil, 'accuracy': accuracy,
                'n_clusters': len(np.unique(labels_test)), 'analysis': analysis
            })
        
        # mostrar tabla resumen
        print(f"\n{'#':<3} {'Algoritmo':<12} {'Config':<10} {'Silhouette':>11} {'Accuracy':>10} {'Clusters':>8}")
        print("-"*70)
        for i, r in enumerate(results, 1):
            print(f"{i:<3} {r['algorithm']:<12} {r['config']:<10} {r['test_sil']:>11.4f} {r['accuracy']:>10.2%} {r['n_clusters']:>8}")
        
        # detalle de pureza por cluster
        print("\n" + "-"*70)
        print("DETALLE POR CLUSTER")
        print("-"*70)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['algorithm']} - {r['config']}")
            avg_purity = np.mean([a['purity'] for a in r['analysis'].values()])
            print(f"    Pureza promedio: {avg_purity:.2%}")
            for c, a in list(r['analysis'].items())[:3]:
                print(f"    Cluster {c}: {a['n']} muestras, clase dominante='{a['dominant_name']}', pureza={a['purity']:.2%}")
        
        # metricas finales
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_sil = np.mean([r['test_sil'] for r in results])
        
        print("\n" + "="*70)
        print("MÉTRICAS FINALES")
        print("="*70)
        print(f"Accuracy promedio:    {avg_acc:.2%}")
        print(f"Silhouette promedio:  {avg_sil:.4f}")
        
        return results

# generar gráficos comparativos
def plot_results(all_results, test_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # gráfico 1: Top 3 en test set
    labels = [f"{r['algorithm']}\n{r['config']}" for r in test_results]
    scores = [r['test_sil'] for r in test_results]
    bars = ax1.bar(range(len(labels)), scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontweight='bold')
    ax1.set_title('Top 3 en Test Set', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels)
    ax1.grid(alpha=0.3, axis='y')
    
    # gráfico 2: Todas las 12 configuraciones en train
    all_labels = [f"{r['algorithm'][:8]}\n{r['config']}" for r in all_results]
    all_scores = [r['score'] for r in all_results]
    colors = ['#1f77b4' if 'K-Means' in r['algorithm'] and '++' not in r['algorithm'] 
              else '#ff7f0e' if '++' in r['algorithm'] else '#2ca02c' for r in all_results]
    bars = ax2.bar(range(len(all_labels)), all_scores, color=colors, alpha=0.7)
    
    # destacar top 3 con borde rojo
    top_idx = [all_results.index(r) for r in sorted(all_results, key=lambda x: x['score'], reverse=True)[:3]]
    for idx in top_idx:
        bars[idx].set_edgecolor('red')
        bars[idx].set_linewidth(2.5)
    
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    ax2.set_title('12 Configuraciones en Train', fontweight='bold', fontsize=14)
    ax2.set_xticks(range(len(all_labels)))
    ax2.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# función principal - ejecuta todo el análisis
def main():
    print("\n" + "="*70)
    print("TAREA 2 - ITEM 1: CLUSTERING")
    print("="*70)
    
    # paso 1: cargar datos
    data = load_data()
    print(f"\nDataset: {data['X_train'].shape[0] + data['X_test'].shape[0]} filas")
    print(f"Train: {data['X_train'].shape} | Test: {data['X_test'].shape}")
    print(f"Clases: {data['n_classes']}")
    
    # paso 2: definir parámetros (4 por algoritmo)
    params = get_parameters(data['n_classes'])
    print(f"\nParámetros K-Means: k = {params['k_values']}")
    print(f"Parámetros MeanShift: quantiles = {params['quantiles']}")
    
    # paso 3: evaluar 12 configuraciones en train
    analyzer = ClusteringAnalysis(data)
    all_results = analyzer.evaluate_all(params)
    
    # paso 4: seleccionar top 3
    top3 = all_results[:3]
    
    # paso 5: evaluar top 3 en test y comparar con etiquetas reales
    test_results = analyzer.evaluate_test(top3)
    
    # paso 6: generar visualización
    print("\nGenerando gráficos...")
    plot_results(all_results, test_results)
    
    print("\n" + "="*70)
    print("COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()