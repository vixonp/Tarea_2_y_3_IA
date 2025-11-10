import numpy as np
import pandas as pd
from data_loader import load_dataset, preprocess_data
from clustering import ClusteringComparison, justify_parameters
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 80)
    print("ACTIVIDAD 1: CLUSTERING (K-Means, K-Means++, MeanShift)")
    print("=" * 80)
    
    # ============================================================================
    # CARGAR Y PREPROCESAR DATOS
    # ============================================================================
    print("\n1. CARGANDO Y PREPROCESANDO DATOS")
    print("-" * 80)
    
    # Cargar dataset
    # Opción 1: Usar un dataset de sklearn
    df = load_dataset()
    
    # Opción 2: Si tienes un archivo CSV, descomenta la siguiente línea:
    # df = load_dataset(file_path='tu_dataset.csv')
    
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {df.columns.tolist()}")
    
    # Seleccionar columna objetivo (ajustar según tu dataset)
    target_column = 'target'  # Cambiar según tu dataset
    
    if target_column not in df.columns:
        # Si no existe 'target', usar la última columna
        target_column = df.columns[-1]
        print(f"Columna objetivo no encontrada. Usando: {target_column}")
    
    print(f"\nDistribución de clases en '{target_column}':")
    print(df[target_column].value_counts())
    
    # Preprocesar datos
    data = preprocess_data(df, target_column, test_size=0.2, random_state=42)
    
    print(f"\nDatos preprocesados:")
    print(f"  Train set: {data['X_train'].shape}")
    print(f"  Test set: {data['X_test'].shape}")
    print(f"  Clases únicas: {len(np.unique(data['y_train']))}")
    
    # ============================================================================
    # PARTE 1: CLUSTERING
    # ============================================================================
    print("\n" + "=" * 80)
    print("PARTE 1: CLUSTERING (K-Means, K-Means++, MeanShift)")
    print("=" * 80)
    
    # Justificar parámetros
    justification = justify_parameters()
    print("\nJustificación de parámetros:")
    for alg, params in justification.items():
        print(f"\n{alg}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Crear comparador de clustering
    clustering_comp = ClusteringComparison(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test']
    )
    
    # Evaluar todas las configuraciones
    all_results = clustering_comp.evaluate_all_configurations()
    
    # Seleccionar top 3 configuraciones
    top_configs = clustering_comp.select_top_configurations(top_n=3)
    
    # Evaluar en conjunto de test
    test_results = clustering_comp.evaluate_top_configurations_on_test(top_configs)
    
    # Análisis de resultados
    print("\n" + "-" * 80)
    print("ANÁLISIS DE RESULTADOS DE CLUSTERING")
    print("-" * 80)
    
    for result in test_results:
        print(f"\n{result['algorithm']} - {result['config']}:")
        print(f"  Silhouette Score (test): {result['test_silhouette']:.4f}")
        print(f"  Precisión en asignación de etiquetas: {result['label_accuracy']:.4f}")
        print(f"  Número de clusters: {result['n_clusters']}")
        
        print(f"\n  Análisis por cluster:")
        for cluster_id, analysis in result['cluster_analysis'].items():
            print(f"    Cluster {cluster_id}:")
            print(f"      Muestras: {analysis['total_samples']}")
            print(f"      Etiqueta dominante: {analysis['dominant_label']}")
            print(f"      Distribución: {analysis['label_distribution']}")
    
    print("\n" + "=" * 80)
    print("CONCLUSIÓN PARTE 1:")
    print("=" * 80)
    print("""
    El procedimiento de asignar etiquetas basado en el cluster dominante puede
    ser razonable si:
    1. Los clusters son coherentes (alta homogeneidad interna)
    2. Hay una etiqueta claramente dominante en cada cluster
    3. El silhouette score es alto (clusters bien definidos)
    
    Sin embargo, este método tiene limitaciones:
    - Asume que muestras similares pertenecen a la misma clase
    - No considera la estructura no lineal de los datos
    - Puede fallar si las clases no son separables por distancia euclidiana
    """)
    
    # ============================================================================
    # GENERAR GRÁFICOS
    # ============================================================================
    try:
        print("\nGenerando gráficos de resultados de clustering...")
        
        # Gráfico de comparación de clustering
        plt.figure(figsize=(12, 6))
        algorithms = [r['algorithm'] for r in test_results]
        scores = [r['test_silhouette'] for r in test_results]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = plt.bar(range(len(algorithms)), scores, color=colors[:len(algorithms)])
        
        # Agregar valores en las barras
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Configuración', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Comparación de Algoritmos de Clustering (Test Set)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(algorithms)), 
                   [f"{r['algorithm']}\n{r['config']}" for r in test_results],
                   rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ Gráfico guardado: clustering_comparison.png")
        
        # Gráfico de todas las configuraciones evaluadas
        plt.figure(figsize=(14, 8))
        all_algorithms = [r['algorithm'] for r in all_results]
        all_configs = [r['config'] for r in all_results]
        all_scores = [r['score'] for r in all_results]
        
        x_pos = np.arange(len(all_results))
        colors_all = ['#1f77b4' if 'K-Means' in alg and '+' not in alg else 
                     '#ff7f0e' if 'K-Means++' in alg else 
                     '#2ca02c' for alg in all_algorithms]
        
        bars = plt.bar(x_pos, all_scores, color=colors_all, alpha=0.7)
        
        # Resaltar las top 3
        top_indices = [all_results.index(config) for config in top_configs]
        for idx in top_indices:
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(3)
        
        plt.xlabel('Configuración', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.title('Todas las Configuraciones de Clustering Evaluadas (Train Set)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(x_pos, [f"{alg}\n{config}" for alg, config in zip(all_algorithms, all_configs)],
                  rotation=45, ha='right', fontsize=9)
        plt.legend([bars[0], bars[1], bars[2], bars[top_indices[0]]], 
                  ['K-Means', 'K-Means++', 'MeanShift', 'Top 3 seleccionadas'],
                  loc='upper right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('clustering_all_configurations.png', dpi=300, bbox_inches='tight')
        print("  ✓ Gráfico guardado: clustering_all_configurations.png")
        
    except Exception as e:
        print(f"  ✗ Error al generar gráficos: {e}")
    
    print("\n" + "=" * 80)
    print("ACTIVIDAD 1 COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()
