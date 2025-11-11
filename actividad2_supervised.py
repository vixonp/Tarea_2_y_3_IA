import numpy as np
import pandas as pd
from data_loader import load_dataset, preprocess_data
from supervised_learning import ParallelModelTraining
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Agregar import de os

def main():
    print("=" * 80)
    print("ACTIVIDAD 2: APRENDIZAJE SUPERVISADO (Regresión Logística y SVM)")
    print("=" * 80)
    
    # ============================================================================
    # CARGAR Y PREPROCESAR DATOS
    # ============================================================================
    print("\n1. CARGANDO Y PREPROCESANDO DATOS")
    print("-" * 80)
    
    # Cargar dataset - Usar la misma lógica que actividad1_clustering.py
    # Buscar archivo CSV en el directorio
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if 'DryBeanDataset.csv' in csv_files:
        df = load_dataset(file_path='DryBeanDataset.csv')
    elif 'Dry_Bean_Dataset.csv' in csv_files:
        df = load_dataset(file_path='Dry_Bean_Dataset.csv')
    elif 'DryBeanDataset_ar.csv' in csv_files:
        df = load_dataset(file_path='DryBeanDataset_ar.csv')
    elif csv_files:
        print(f"Archivos CSV encontrados: {csv_files}")
        print(f"Usando el primer archivo encontrado: {csv_files[0]}")
        df = load_dataset(file_path=csv_files[0])
    else:
        print("No se encontró archivo CSV. Usando dataset sintético...")
        df = load_dataset()
    
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {df.columns.tolist()}")
    
    # Seleccionar columna objetivo - Usar la misma lógica que actividad1_clustering.py
    # Intentar diferentes nombres comunes de columna objetivo
    target_column = None
    
    # Priorizar nombres comunes
    possible_target_names = ['Class', 'class', 'target', 'Target', 'label', 'Label', 'y', 'Y']
    
    for name in possible_target_names:
        if name in df.columns:
            target_column = name
            break
    
    # Si no se encontró ninguna, usar la última columna
    if target_column is None:
        target_column = df.columns[-1]
        print(f"Columna objetivo no encontrada en nombres comunes. Usando: {target_column}")
    else:
        print(f"Columna objetivo detectada: {target_column}")
    
    print(f"\nDistribución de clases en '{target_column}':")
    print(df[target_column].value_counts())
    
    # Preprocesar datos
    data = preprocess_data(df, target_column, test_size=0.2, random_state=42)
    
    print(f"\nDatos preprocesados:")
    print(f"  Train set: {data['X_train'].shape}")
    print(f"  Test set: {data['X_test'].shape}")
    print(f"  Clases únicas: {len(np.unique(data['y_train']))}")
    
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
    # GENERAR GRÁFICOS
    # ============================================================================
    try:
        print("\nGenerando gráficos de resultados de aprendizaje supervisado...")
        
        # Gráfico de evolución del entrenamiento
        plt.figure(figsize=(14, 8))
        for model in trainer.models:
            if model['scores']:
                epochs = [s['epoch'] for s in model['scores']]
                f1_scores = [s['f1_score'] for s in model['scores']]
                accuracy_scores = [s['accuracy'] for s in model['scores']]
                
                # Determinar estilo según si es uno de los mejores
                is_best = any(m['name'] == model['name'] for m in best_models)
                linestyle = '-' if is_best else '--'
                linewidth = 2.5 if is_best else 1.5
                alpha = 1.0 if is_best else 0.6
                
                plt.plot(epochs, f1_scores, label=f"{model['name']} (F1)", 
                        marker='o', linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Evolución del Entrenamiento - F1 Score por Época', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_evolution_f1.png', dpi=300, bbox_inches='tight')
        print("  ✓ Gráfico guardado: training_evolution_f1.png")
        
        # Gráfico de evolución de accuracy
        plt.figure(figsize=(14, 8))
        for model in trainer.models:
            if model['scores']:
                epochs = [s['epoch'] for s in model['scores']]
                accuracy_scores = [s['accuracy'] for s in model['scores']]
                
                is_best = any(m['name'] == model['name'] for m in best_models)
                linestyle = '-' if is_best else '--'
                linewidth = 2.5 if is_best else 1.5
                alpha = 1.0 if is_best else 0.6
                
                plt.plot(epochs, accuracy_scores, label=f"{model['name']} (Accuracy)", 
                        marker='s', linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Evolución del Entrenamiento - Accuracy por Época', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_evolution_accuracy.png', dpi=300, bbox_inches='tight')
        print("  ✓ Gráfico guardado: training_evolution_accuracy.png")
        
        # Gráfico de comparación de resultados en test set
        plt.figure(figsize=(12, 6))
        model_names = [r['name'] for r in test_results_supervised]
        test_accuracies = [r['test_accuracy'] for r in test_results_supervised]
        test_f1 = [r['test_f1_weighted'] for r in test_results_supervised]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, test_accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = plt.bar(x + width/2, test_f1, width, label='F1 Score (weighted)', alpha=0.8)
        
        # Agregar valores en las barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Modelo', fontsize=12)
        plt.ylabel('Métrica', fontsize=12)
        plt.title('Comparación de Mejores Modelos en Test Set', fontsize=14, fontweight='bold')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('supervised_test_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ Gráfico guardado: supervised_test_comparison.png")
        
    except Exception as e:
        print(f"  ✗ Error al generar gráficos: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ACTIVIDAD 2 COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    main()
