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

# Cargar modelos e historiales
model_base = keras.models.load_model(os.path.join(SAVE_DIR, 'modelo_no_dropout.keras'))
model_dropout = keras.models.load_model(os.path.join(SAVE_DIR, 'modelo_dropout.keras'))

with open(os.path.join(SAVE_DIR, 'history_no_dropout.pkl'), 'rb') as f:
    history_base = pickle.load(f)
with open(os.path.join(SAVE_DIR, 'history_dropout.pkl'), 'rb') as f:
    history_dropout = pickle.load(f)

# Mostrar arquitecturas
print("ARQUITECTURA MODELO SIN DROPOUT")
model_base.summary()
print("\nARQUITECTURA MODELO CON DROPOUT")
model_dropout.summary()

# Preparar conjunto de prueba
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, 'testing'),
    image_size=(64, 64),
    batch_size=32,
    label_mode='int',
    shuffle=False,
    verbose=0
)
class_names = test_ds_raw.class_names
test_ds = test_ds_raw.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Evaluación en conjunto de prueba
print("\nEvaluando modelos...")
test_loss_base, test_acc_base = model_base.evaluate(test_ds, verbose=1)
test_loss_dropout, test_acc_dropout = model_dropout.evaluate(test_ds, verbose=1)

# Predicciones
y_pred_base = np.argmax(model_base.predict(test_ds, verbose=0), axis=1)
y_pred_dropout = np.argmax(model_dropout.predict(test_ds, verbose=0), axis=1)

# F1-Score
f1_base = f1_score(y_true, y_pred_base, average='weighted')
f1_dropout = f1_score(y_true, y_pred_dropout, average='weighted')

# Classification Reports
print("\nCLASSIFICATION REPORT - MODELO SIN DROPOUT")
print(classification_report(y_true, y_pred_base, target_names=class_names, digits=4))
print("\nCLASSIFICATION REPORT - MODELO CON DROPOUT")
print(classification_report(y_true, y_pred_dropout, target_names=class_names, digits=4))

# Curvas de entrenamiento
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history_base['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history_base['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_title('Modelo SIN Dropout - Loss')
axes[0, 0].set_xlabel('Epoca')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history_base['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 1].plot(history_base['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 1].set_title('Modelo SIN Dropout - Accuracy')
axes[0, 1].set_xlabel('Epoca')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history_dropout['loss'], label='Training Loss', linewidth=2)
axes[1, 0].plot(history_dropout['val_loss'], label='Validation Loss', linewidth=2)
axes[1, 0].set_title('Modelo CON Dropout - Loss')
axes[1, 0].set_xlabel('Epoca')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history_dropout['accuracy'], label='Training Accuracy', linewidth=2)
axes[1, 1].plot(history_dropout['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1, 1].set_title('Modelo CON Dropout - Accuracy')
axes[1, 1].set_xlabel('Epoca')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'curvas_entrenamiento.png'), dpi=150, bbox_inches='tight')
plt.close()

# Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_base = confusion_matrix(y_true, y_pred_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Matriz de Confusion - Modelo SIN Dropout')
axes[0].set_ylabel('Etiqueta Real')
axes[0].set_xlabel('Etiqueta Predicha')

cm_dropout = confusion_matrix(y_true, y_pred_dropout)
sns.heatmap(cm_dropout, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Matriz de Confusion - Modelo CON Dropout')
axes[1].set_ylabel('Etiqueta Real')
axes[1].set_xlabel('Etiqueta Predicha')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'matrices_confusion.png'), dpi=150, bbox_inches='tight')
plt.close()

# Tabla comparativa
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

final_train_acc_base = history_base['accuracy'][-1]
final_val_acc_base = history_base['val_accuracy'][-1]
final_train_acc_dropout = history_dropout['accuracy'][-1]
final_val_acc_dropout = history_dropout['val_accuracy'][-1]
gap_base = final_train_acc_base - final_val_acc_base
gap_dropout = final_train_acc_dropout - final_val_acc_dropout

tabla_data = [
    ['Metrica', 'Modelo SIN Dropout', 'Modelo CON Dropout', 'Diferencia'],
    ['Training Accuracy', f'{final_train_acc_base:.4f}', f'{final_train_acc_dropout:.4f}',
     f'{(final_train_acc_dropout - final_train_acc_base)*100:+.2f}%'],
    ['Validation Accuracy', f'{final_val_acc_base:.4f}', f'{final_val_acc_dropout:.4f}',
     f'{(final_val_acc_dropout - final_val_acc_base)*100:+.2f}%'],
    ['Test Accuracy', f'{test_acc_base:.4f}', f'{test_acc_dropout:.4f}',
     f'{(test_acc_dropout - test_acc_base)*100:+.2f}%'],
    ['Test Loss', f'{test_loss_base:.4f}', f'{test_loss_dropout:.4f}',
     f'{(test_loss_dropout - test_loss_base):+.4f}'],
    ['F1-Score (Test)', f'{f1_base:.4f}', f'{f1_dropout:.4f}',
     f'{(f1_dropout - f1_base):+.4f}'],
    ['Overfitting Gap*', f'{gap_base:.4f}', f'{gap_dropout:.4f}',
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

plt.title('Comparacion de Modelos', fontsize=16, fontweight='bold', pad=20)
plt.figtext(0.5, 0.02, '*Overfitting Gap = Training Accuracy - Validation Accuracy (menor es mejor)', 
            ha='center', fontsize=9, style='italic')
plt.savefig(os.path.join(SAVE_DIR, 'tabla_comparativa.png'), dpi=150, bbox_inches='tight')
plt.close()

# Resumen final
print("\n" + "="*65)
print("RESUMEN COMPARATIVO")
print("="*65)
print(f"{'Modelo':<20} {'Accuracy':<15} {'F1-Score':<15} {'Loss':<15}")
print("-" * 65)
print(f"{'SIN Dropout':<20} {test_acc_base:<15.4f} {f1_base:<15.4f} {test_loss_base:<15.4f}")
print(f"{'CON Dropout':<20} {test_acc_dropout:<15.4f} {f1_dropout:<15.4f} {test_loss_dropout:<15.4f}")
print("-" * 65)
print(f"Overfitting Gap SIN Dropout: {gap_base:.4f}")
print(f"Overfitting Gap CON Dropout: {gap_dropout:.4f}")
print(f"Reduccion de Overfitting: {(gap_base - gap_dropout):.4f}")
print("="*65)
print(f"Archivos guardados en {SAVE_DIR}/")