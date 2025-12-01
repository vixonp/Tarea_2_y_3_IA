"""
TAREA 3 - EVALUACIÓN DE MODELOS
Evalúa modelo base vs modelo con dropout en conjunto de prueba
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import pickle
import os

BASE_DIR = 'imagenes_divididas'
SAVE_DIR = './modelos_entrenados'

print("="*80)
print("EVALUACIÓN DE MODELOS - CONJUNTO DE PRUEBA")
print("="*80)

# ==============================================================================
# CARGAR MODELOS E HISTORIALES
# ==============================================================================

print("\n[1] CARGANDO MODELOS Y DATOS")
print("-"*80)

if not os.path.exists(SAVE_DIR):
    print(f"❌ ERROR: No existe la carpeta {SAVE_DIR}")
    print("Primero debes ejecutar entrenar.py")
    exit()

# Cargar modelos
try:
    model_base = keras.models.load_model(os.path.join(SAVE_DIR, 'modelo_no_dropout.keras'))
    print("✅ Modelo sin dropout cargado")
except:
    print("❌ ERROR: No se pudo cargar modelo_no_dropout.keras")
    exit()

try:
    model_dropout = keras.models.load_model(os.path.join(SAVE_DIR, 'modelo_dropout.keras'))
    print("✅ Modelo con dropout cargado")
except:
    print("❌ ERROR: No se pudo cargar modelo_dropout.keras")
    exit()

# Cargar historiales
try:
    with open(os.path.join(SAVE_DIR, 'history_no_dropout.pkl'), 'rb') as f:
        history_base = pickle.load(f)
    print("✅ Historial sin dropout cargado")
except:
    print("❌ ERROR: No se pudo cargar history_no_dropout.pkl")
    exit()

try:
    with open(os.path.join(SAVE_DIR, 'history_dropout.pkl'), 'rb') as f:
        history_dropout = pickle.load(f)
    print("✅ Historial con dropout cargado")
except:
    print("❌ ERROR: No se pudo cargar history_dropout.pkl")
    exit()

print("\n✅ Todos los archivos cargados exitosamente")

# ==============================================================================
# ARQUITECTURAS
# ==============================================================================

print("\n" + "="*80)
print("ARQUITECTURA MODELO SIN DROPOUT")
print("="*80)
model_base.summary()

print("\n" + "="*80)
print("ARQUITECTURA MODELO CON DROPOUT")
print("="*80)
model_dropout.summary()

# ==============================================================================
# PREPARAR CONJUNTO DE PRUEBA
# ==============================================================================

print("\n" + "="*80)
print("[2] PREPARANDO CONJUNTO DE PRUEBA")
print("="*80)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, 'testing'),
    image_size=(64, 64),
    batch_size=32,
    label_mode='int',
    shuffle=False,
    verbose=0
)

# Obtener nombres de clases ANTES de optimizar
class_names = test_ds_raw.class_names
print(f"✅ Clases encontradas: {class_names}")

# Ahora optimizar
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds_raw.cache().prefetch(buffer_size=AUTOTUNE)

# Extraer etiquetas verdaderas
y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

print(f"✅ Total de imágenes de prueba: {len(y_true)}")

# ==============================================================================
# EVALUACIÓN EN CONJUNTO DE PRUEBA
# ==============================================================================

print("\n" + "="*80)
print("[3] EVALUACIÓN EN CONJUNTO DE PRUEBA")
print("="*80)

print("\nEvaluando Modelo SIN Dropout...")
test_loss_base, test_acc_base = model_base.evaluate(test_ds, verbose=1)

print("\nEvaluando Modelo CON Dropout...")
test_loss_dropout, test_acc_dropout = model_dropout.evaluate(test_ds, verbose=1)

print("\n" + "-"*80)
print("RESULTADOS EN TEST SET:")
print("-"*80)
print(f"\nModelo SIN Dropout:")
print(f"  Test Loss:     {test_loss_base:.4f}")
print(f"  Test Accuracy: {test_acc_base:.4f} ({test_acc_base*100:.2f}%)")

print(f"\nModelo CON Dropout:")
print(f"  Test Loss:     {test_loss_dropout:.4f}")
print(f"  Test Accuracy: {test_acc_dropout:.4f} ({test_acc_dropout*100:.2f}%)")

# ==============================================================================
# PREDICCIONES Y MÉTRICAS DETALLADAS
# ==============================================================================

print("\n" + "="*80)
print("[4] GENERANDO PREDICCIONES Y MÉTRICAS")
print("="*80)

# Predicciones
print("\nGenerando predicciones Modelo SIN Dropout...")
y_pred_base = model_base.predict(test_ds, verbose=0)
y_pred_classes_base = np.argmax(y_pred_base, axis=1)

print("Generando predicciones Modelo CON Dropout...")
y_pred_dropout = model_dropout.predict(test_ds, verbose=0)
y_pred_classes_dropout = np.argmax(y_pred_dropout, axis=1)

# F1-Score
f1_base = f1_score(y_true, y_pred_classes_base, average='weighted')
f1_dropout = f1_score(y_true, y_pred_classes_dropout, average='weighted')

print(f"\n✅ F1-Score SIN Dropout: {f1_base:.4f}")
print(f"✅ F1-Score CON Dropout: {f1_dropout:.4f}")

# Classification Reports
print("\n" + "="*80)
print("CLASSIFICATION REPORT - MODELO SIN DROPOUT")
print("="*80)
print(classification_report(y_true, y_pred_classes_base, target_names=class_names, digits=4))

print("\n" + "="*80)
print("CLASSIFICATION REPORT - MODELO CON DROPOUT")
print("="*80)
print(classification_report(y_true, y_pred_classes_dropout, target_names=class_names, digits=4))

# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

print("\n" + "="*80)
print("[5] GENERANDO VISUALIZACIONES")
print("="*80)

# 1. CURVAS DE ENTRENAMIENTO
print("\nGenerando curvas de entrenamiento...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Modelo Base - Loss
axes[0, 0].plot(history_base['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
axes[0, 0].plot(history_base['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
axes[0, 0].set_title('Modelo SIN Dropout - Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Modelo Base - Accuracy
axes[0, 1].plot(history_base['accuracy'], label='Training Accuracy', linewidth=2, color='#2E86AB')
axes[0, 1].plot(history_base['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#A23B72')
axes[0, 1].set_title('Modelo SIN Dropout - Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Modelo Dropout - Loss
axes[1, 0].plot(history_dropout['loss'], label='Training Loss', linewidth=2, color='#06A77D')
axes[1, 0].plot(history_dropout['val_loss'], label='Validation Loss', linewidth=2, color='#D5573B')
axes[1, 0].set_title('Modelo CON Dropout - Loss', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Época')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Modelo Dropout - Accuracy
axes[1, 1].plot(history_dropout['accuracy'], label='Training Accuracy', linewidth=2, color='#06A77D')
axes[1, 1].plot(history_dropout['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#D5573B')
axes[1, 1].set_title('Modelo CON Dropout - Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Época')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'curvas_entrenamiento.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Guardado: curvas_entrenamiento.png")

# 2. MATRICES DE CONFUSIÓN
print("Generando matrices de confusión...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_base = confusion_matrix(y_true, y_pred_classes_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0],
            cbar_kws={'label': 'Número de predicciones'})
axes[0].set_title('Matriz de Confusión - Modelo SIN Dropout', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Etiqueta Real')
axes[0].set_xlabel('Etiqueta Predicha')

cm_dropout = confusion_matrix(y_true, y_pred_classes_dropout)
sns.heatmap(cm_dropout, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[1],
            cbar_kws={'label': 'Número de predicciones'})
axes[1].set_title('Matriz de Confusión - Modelo CON Dropout', fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Etiqueta Real')
axes[1].set_xlabel('Etiqueta Predicha')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'matrices_confusion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Guardado: matrices_confusion.png")

# 3. TABLA COMPARATIVA
print("Generando tabla comparativa...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Métricas finales
final_train_acc_base = history_base['accuracy'][-1]
final_val_acc_base = history_base['val_accuracy'][-1]
final_train_loss_base = history_base['loss'][-1]
final_val_loss_base = history_base['val_loss'][-1]

final_train_acc_dropout = history_dropout['accuracy'][-1]
final_val_acc_dropout = history_dropout['val_accuracy'][-1]
final_train_loss_dropout = history_dropout['loss'][-1]
final_val_loss_dropout = history_dropout['val_loss'][-1]

gap_base = final_train_acc_base - final_val_acc_base
gap_dropout = final_train_acc_dropout - final_val_acc_dropout

tabla_data = [
    ['Métrica', 'Modelo SIN Dropout', 'Modelo CON Dropout', 'Diferencia'],
    ['', '', '', ''],
    ['Training Accuracy', f'{final_train_acc_base:.4f} ({final_train_acc_base*100:.2f}%)', 
     f'{final_train_acc_dropout:.4f} ({final_train_acc_dropout*100:.2f}%)',
     f'{(final_train_acc_dropout - final_train_acc_base)*100:+.2f}%'],
    ['Validation Accuracy', f'{final_val_acc_base:.4f} ({final_val_acc_base*100:.2f}%)', 
     f'{final_val_acc_dropout:.4f} ({final_val_acc_dropout*100:.2f}%)',
     f'{(final_val_acc_dropout - final_val_acc_base)*100:+.2f}%'],
    ['Test Accuracy', f'{test_acc_base:.4f} ({test_acc_base*100:.2f}%)', 
     f'{test_acc_dropout:.4f} ({test_acc_dropout*100:.2f}%)',
     f'{(test_acc_dropout - test_acc_base)*100:+.2f}%'],
    ['', '', '', ''],
    ['Training Loss', f'{final_train_loss_base:.4f}', 
     f'{final_train_loss_dropout:.4f}',
     f'{(final_train_loss_dropout - final_train_loss_base):+.4f}'],
    ['Validation Loss', f'{final_val_loss_base:.4f}', 
     f'{final_val_loss_dropout:.4f}',
     f'{(final_val_loss_dropout - final_val_loss_base):+.4f}'],
    ['Test Loss', f'{test_loss_base:.4f}', 
     f'{test_loss_dropout:.4f}',
     f'{(test_loss_dropout - test_loss_base):+.4f}'],
    ['', '', '', ''],
    ['F1-Score (Test)', f'{f1_base:.4f}', 
     f'{f1_dropout:.4f}',
     f'{(f1_dropout - f1_base):+.4f}'],
    ['', '', '', ''],
    ['Overfitting Gap*', f'{gap_base:.4f} ({gap_base*100:.2f}%)', 
     f'{gap_dropout:.4f} ({gap_dropout*100:.2f}%)',
     f'{(gap_dropout - gap_base)*100:+.2f}%'],
]

table = ax.table(cellText=tabla_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

for row in [1, 5, 9, 11]:
    for col in range(4):
        table[(row, col)].set_facecolor('#E8E8E8')

plt.title('Comparación Detallada de Modelos', fontsize=16, fontweight='bold', pad=20)
plt.figtext(0.5, 0.02, '*Overfitting Gap = Training Accuracy - Validation Accuracy (menor es mejor)', 
            ha='center', fontsize=9, style='italic')

plt.savefig(os.path.join(SAVE_DIR, 'tabla_comparativa.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Guardado: tabla_comparativa.png")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================

print("\n" + "="*80)
print("RESUMEN COMPARATIVO")
print("="*80)

print("\nMÉTRICAS EN TEST SET:")
print(f"{'Modelo':<20} {'Accuracy':<15} {'F1-Score':<15} {'Loss':<15}")
print("-" * 65)
print(f"{'SIN Dropout':<20} {test_acc_base:<15.4f} {f1_base:<15.4f} {test_loss_base:<15.4f}")
print(f"{'CON Dropout':<20} {test_acc_dropout:<15.4f} {f1_dropout:<15.4f} {test_loss_dropout:<15.4f}")

print("\nANÁLISIS DE OVERFITTING:")
print(f"  Modelo SIN Dropout - Gap (Train-Val): {gap_base:.4f} ({gap_base*100:.2f}%)")
print(f"  Modelo CON Dropout - Gap (Train-Val): {gap_dropout:.4f} ({gap_dropout*100:.2f}%)")
print(f"  Reducción de Overfitting: {(gap_base - gap_dropout):.4f} ({(gap_base - gap_dropout)*100:.2f}%)")

print("\nCONCLUSIONES:")
if test_acc_dropout > test_acc_base:
    mejora = (test_acc_dropout - test_acc_base) * 100
    print(f"  ✅ El modelo con Dropout supera al base en {mejora:.2f}% accuracy")
    print(f"  ✅ Dropout mejora la generalización del modelo")
else:
    print(f"  ⚠️  El modelo base tuvo mejor accuracy en test set")
    
if gap_dropout < gap_base:
    reduccion = ((gap_base - gap_dropout) / gap_base) * 100
    print(f"  ✅ Dropout reduce el overfitting en {reduccion:.1f}%")
else:
    print(f"  ⚠️  El overfitting no se redujo significativamente")

if f1_dropout > f1_base:
    print(f"  ✅ F1-Score mejor con Dropout (mejor balance precision/recall)")
else:
    print(f"  ⚠️  F1-Score mejor sin Dropout")

print("\n" + "="*80)
print("EVALUACIÓN COMPLETADA")
print("="*80)

print("\nArchivos generados:")
print(f"  📊 {SAVE_DIR}/curvas_entrenamiento.png")
print(f"  📊 {SAVE_DIR}/matrices_confusion.png")
print(f"  📊 {SAVE_DIR}/tabla_comparativa.png")

print("\n✅ Todos los gráficos y análisis están listos para el video explicativo")
print("="*80)