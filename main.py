import numpy as np
import pandas as pd
from data_loader import load_dataset, preprocess_data
from clustering import ClusteringComparison, justify_parameters
from supervised_learning import ParallelModelTraining
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=" * 80)
    print("TAREA 2: COMPARACIÓN DE ALGORITMOS DE APRENDIZAJE")
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
    # PARTE 2: APRENDIZAJE SUPERVISADO
    # ============================================================================
    print("\n" + "=" * 80)
    print("PARTE 2: APRENDIZAJE SUPERVISADO (Regresión Logística y SVM)")
    print("=" * 80)
    
    # Crear entrenador paralelo
    trainer = ParallelModelTraining(
        data['X_train'], data['X_test'],
        data['y_train'], data['y_test'],
        config_path='config.json'
    )
    
    # Entrenar modelos en paralelo
    final_models = trainer.train_all_models_parallel(epochs=50, eval_interval=5)
    
    # Seleccionar mejores modelos
    best_models = trainer.select_best_models(top_n=2)
    
    # Evaluar en conjunto de test
    test_results_supervised = trainer.evaluate_on_test_set(best_models)
    
    # Analizar hiperparámetros
    trainer.analyze_hyperparameters(test_results_supervised)
    
    print("\n" + "=" * 80)
    print("CONCLUSIÓN PARTE 2:")
    print("=" * 80)
    print("""
    Análisis de hiperparámetros:
    
    Regresión Logística:
    - Learning rate: Valores muy altos pueden causar inestabilidad, valores muy
      bajos pueden hacer el entrenamiento muy lento
    - Batch size: Tamaños pequeños pueden ayudar a escapar mínimos locales,
      tamaños grandes pueden acelerar el entrenamiento
    - C (regularization): Valores altos = menos regularización (mayor complejidad),
      valores bajos = más regularización (menor complejidad, mejor generalización)
    
    SVM:
    - C (regularization): Similar a Regresión Logística
    - Kernel: RBF es bueno para datos no lineales, lineal para datos lineales
    - Gamma: Controla la influencia de cada ejemplo. Valores altos = influencia
      local, valores bajos = influencia global
    """)
    
    # ============================================================================
    # GENERAR GRÁFICOS (OPCIONAL)
    # ============================================================================
    try:
        print("\nGenerando gráficos de resultados...")
        
        # Gráfico de evolución del entrenamiento
        plt.figure(figsize=(12, 6))
        for model in trainer.models:
            if model['scores']:
                epochs = [s['epoch'] for s in model['scores']]
                f1_scores = [s['f1_score'] for s in model['scores']]
                plt.plot(epochs, f1_scores, label=model['name'], marker='o')
        
        plt.xlabel('Época')
        plt.ylabel('F1 Score')
        plt.title('Evolución del Entrenamiento (F1 Score)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_evolution.png', dpi=300)
        print("  Gráfico guardado: training_evolution.png")
        
        # Gráfico de comparación de clustering
        plt.figure(figsize=(10, 6))
        algorithms = [r['algorithm'] for r in test_results]
        scores = [r['test_silhouette'] for r in test_results]
        plt.bar(range(len(algorithms)), scores)
        plt.xlabel('Configuración')
        plt.ylabel('Silhouette Score')
        plt.title('Comparación de Algoritmos de Clustering (Test Set)')
        plt.xticks(range(len(algorithms)), 
                   [f"{r['algorithm']}\n{r['config']}" for r in test_results],
                   rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('clustering_comparison.png', dpi=300)
        print("  Gráfico guardado: clustering_comparison.png")
        
    except Exception as e:
        print(f"  Error al generar gráficos: {e}")
    
    print("\n" + "=" * 80)
    print("EJECUCIÓN COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()