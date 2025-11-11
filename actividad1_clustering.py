import numpy as np
import pandas as pd
from data_loader import load_dataset, preprocess_data
from clustering import ClusteringComparison, justify_parameters
import matplotlib.pyplot as plt
import os

def main():
    print("=" * 80)
    print("CLUSTERING: K-Means, K-Means++, MeanShift")
    print("=" * 80)
    
    # Cargar datos
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if 'DryBeanDataset.csv' in csv_files:
        df = load_dataset(file_path='DryBeanDataset.csv')
    
    print(f"\nDataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    target_column = 'Class' if 'Class' in df.columns else df.columns[-1]
    print(f"Distribución de clases:\n{df[target_column].value_counts()}")
    
    # Preprocesar
    data = preprocess_data(df, target_column, test_size=0.2, random_state=42)
    print(f"\nTrain: {data['X_train'].shape} | Test: {data['X_test'].shape}")
    print(f"Clases únicas: {len(np.unique(data['y_train']))}")
    
    # Justificación de parámetros
    print("\n" + "-" * 80)
    justification = justify_parameters()
    print("Justificación de parámetros:")
    for alg, params in justification.items():
        print(f"\n{alg}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Ejecutar clustering
    print("\n" + "=" * 80)
    clustering_comp = ClusteringComparison(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test']
    )
    
    all_results = clustering_comp.evaluate_all_configurations()
    top_configs = clustering_comp.select_top_configurations(top_n=3)
    test_results = clustering_comp.evaluate_top_configurations_on_test(top_configs)
    
    # Análisis de resultados
    print("\n" + "-" * 80)
    print("RESULTADOS EN TEST SET")
    print("-" * 80)
    
    for result in test_results:
        print(f"\n{result['algorithm']} - {result['config']}:")
        print(f"  Silhouette (test): {result['test_silhouette']:.4f}")
        print(f"  Precisión etiquetas: {result['label_accuracy']:.4f}")
        print(f"  Clusters: {result['n_clusters']}")
        
        print(f"  Análisis por cluster:")
        for cluster_id, analysis in result['cluster_analysis'].items():
            print(f"    Cluster {cluster_id}: {analysis['total_samples']} muestras, "
                  f"dominante={analysis['dominant_label']}, dist={analysis['label_distribution']}")
    
    # Gráficos
    print("\nGenerando gráficos...")
    
    # Comparación top 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    algorithms = [r['algorithm'] for r in test_results]
    scores = [r['test_silhouette'] for r in test_results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars1 = ax1.bar(range(len(algorithms)), scores, color=colors[:len(algorithms)])
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Configuración', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('Top 3 Configuraciones (Test Set)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels([f"{r['algorithm']}\n{r['config']}" for r in test_results],
                        rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Todas las configuraciones
    all_algorithms = [r['algorithm'] for r in all_results]
    all_scores = [r['score'] for r in all_results]
    x_pos = np.arange(len(all_results))
    
    colors_all = ['#1f77b4' if 'K-Means' in alg and '++' not in alg else 
                  '#ff7f0e' if '++' in alg else '#2ca02c' for alg in all_algorithms]
    
    bars2 = ax2.bar(x_pos, all_scores, color=colors_all, alpha=0.7)
    
    top_indices = [all_results.index(config) for config in top_configs]
    for idx in top_indices:
        bars2[idx].set_edgecolor('red')
        bars2[idx].set_linewidth(3)
    
    ax2.set_xlabel('Configuración', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Todas las Configuraciones (Train Set)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{r['algorithm']}\n{r['config']}" for r in all_results],
                        rotation=45, ha='right', fontsize=8)
    ax2.legend(['K-Means', 'K-Means++', 'MeanShift', 'Top 3'], loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("COMPLETADO")
    print("=" * 80)

if __name__ == "__main__":
    main()