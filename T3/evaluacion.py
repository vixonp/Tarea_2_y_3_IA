import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import pickle
import os

BASE_DIR = 'imagenes_divididas'
SAVE_DIR = './modelos_entrenados'

print("="*60)
print("CARGANDO MODELOS Y DATOS")
print("="*60)

if not os.path.exists(SAVE_DIR):
    print(f"ERROR: No existe la carpeta {SAVE_DIR}")
    print("Primero debes ejecutar entrenar.py")
    exit()

model_base = keras.models.load_model(os.path.join(SAVE_DIR, 'modelo_base.h5'))
model_dropout = keras.models.load_model(os.path.join(SAVE_DIR, 'modelo_dropout.h5'))

with open(os.path.join(SAVE_DIR, 'history_base.pkl'), 'rb') as f:
    history_base = pickle.load(f)
with open(os.path.join(SAVE_DIR, 'history_dropout.pkl'), 'rb') as f:
    history_dropout = pickle.load(f)
with open(os.path.join(SAVE_DIR, 'best_hyperparams.pkl'), 'rb') as f:
    best_params = pickle.load(f)
with open(os.path.join(SAVE_DIR, 'dropout_info.pkl'), 'rb') as f:
    dropout_info = pickle.load(f)

print("Archivos cargados exitosamente.")

print("\n" + "="*60)
print("HIPERPARAMETROS UTILIZADOS")
print("="*60)
print(f"Batch Size: {best_params['batch_size']}")
print(f"Epochs: {best_params['epochs']}")
print(f"Learning Rate: {best_params['learning_rate']}")
print(f"Kernel Size: {best_params['kernel_size']}")
print(f"Architecture: {best_params['architecture']}")
print(f"Dropout Rate: {dropout_info['dropout_rate']}")

print("\n" + "="*60)
print("ARQUITECTURA MODELO BASE")
print("="*60)
model_base.summary()

print("\n" + "="*60)
print("ARQUITECTURA MODELO CON DROPOUT")
print("="*60)
model_dropout.summary()

print("\n" + "="*60)
print("PREPARANDO CONJUNTO DE PRUEBA")
print("="*60)

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'testing'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"Imagenes de prueba: {test_generator.samples}")
print(f"Clases: {list(test_generator.class_indices.keys())}")

print("\n" + "="*60)
print("EVALUACION EN CONJUNTO DE PRUEBA")
print("="*60)

test_loss_base, test_acc_base = model_base.evaluate(test_generator, verbose=1)
test_generator.reset()
test_loss_dropout, test_acc_dropout = model_dropout.evaluate(test_generator, verbose=1)

print(f"\nModelo Base:")
print(f"  Test Loss: {test_loss_base:.4f}")
print(f"  Test Accuracy: {test_acc_base:.4f} ({test_acc_base*100:.2f}%)")

print(f"\nModelo con Dropout:")
print(f"  Test Loss: {test_loss_dropout:.4f}")
print(f"  Test Accuracy: {test_acc_dropout:.4f} ({test_acc_dropout*100:.2f}%)")

print("\n" + "="*60)
print("GENERANDO PREDICCIONES")
print("="*60)

test_generator.reset()
y_pred_base = model_base.predict(test_generator, verbose=1)
test_generator.reset()
y_pred_dropout = model_dropout.predict(test_generator, verbose=1)

y_pred_classes_base = np.argmax(y_pred_base, axis=1)
y_pred_classes_dropout = np.argmax(y_pred_dropout, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

f1_base = f1_score(y_true, y_pred_classes_base, average='weighted')
f1_dropout = f1_score(y_true, y_pred_classes_dropout, average='weighted')

print(f"\nF1-Score Base: {f1_base:.4f}")
print(f"F1-Score Dropout: {f1_dropout:.4f}")

print("\n" + "="*60)
print("CLASSIFICATION REPORT - MODELO BASE")
print("="*60)
print(classification_report(y_true, y_pred_classes_base, target_names=class_names, digits=4))

print("\n" + "="*60)
print("CLASSIFICATION REPORT - MODELO CON DROPOUT")
print("="*60)
print(classification_report(y_true, y_pred_classes_dropout, target_names=class_names, digits=4))

print("\n" + "="*60)
print("GENERANDO VISUALIZACIONES")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

axes[0, 0].plot(history_base['loss'], label='Training Loss', linewidth=2, color='#2E86AB')
axes[0, 0].plot(history_base['val_loss'], label='Validation Loss', linewidth=2, color='#A23B72')
axes[0, 0].set_title('Modelo Base - Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoca')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history_base['accuracy'], label='Training Accuracy', linewidth=2, color='#2E86AB')
axes[0, 1].plot(history_base['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#A23B72')
axes[0, 1].set_title('Modelo Base - Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoca')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history_dropout['loss'], label='Training Loss', linewidth=2, color='#06A77D')
axes[1, 0].plot(history_dropout['val_loss'], label='Validation Loss', linewidth=2, color='#D5573B')
axes[1, 0].set_title('Modelo con Dropout - Loss', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoca')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history_dropout['accuracy'], label='Training Accuracy', linewidth=2, color='#06A77D')
axes[1, 1].plot(history_dropout['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#D5573B')
axes[1, 1].set_title('Modelo con Dropout - Accuracy', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoca')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'curvas_entrenamiento.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: curvas_entrenamiento.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_base = confusion_matrix(y_true, y_pred_classes_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0],
            cbar_kws={'label': 'Numero de predicciones'})
axes[0].set_title('Matriz de Confusion - Modelo Base', fontsize=14, fontweight='bold', pad=15)
axes[0].set_ylabel('Etiqueta Real')
axes[0].set_xlabel('Etiqueta Predicha')

cm_dropout = confusion_matrix(y_true, y_pred_classes_dropout)
sns.heatmap(cm_dropout, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[1],
            cbar_kws={'label': 'Numero de predicciones'})
axes[1].set_title('Matriz de Confusion - Modelo con Dropout', fontsize=14, fontweight='bold', pad=15)
axes[1].set_ylabel('Etiqueta Real')
axes[1].set_xlabel('Etiqueta Predicha')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'matrices_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: matrices_confusion.png")

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

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
    ['Metrica', 'Modelo Base', 'Modelo con Dropout', 'Diferencia'],
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

plt.title('Comparacion Detallada de Modelos', fontsize=16, fontweight='bold', pad=20)
plt.figtext(0.5, 0.02, '*Overfitting Gap = Training Accuracy - Validation Accuracy (menor es mejor)', 
            ha='center', fontsize=9, style='italic')

plt.savefig(os.path.join(SAVE_DIR, 'tabla_comparativa.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: tabla_comparativa.png")

print("\n" + "="*80)
print("RESUMEN COMPARATIVO")
print("="*80)

print("\nMETRICAS EN TEST SET:")
print(f"{'Modelo':<20} {'Accuracy':<15} {'F1-Score':<15} {'Loss':<15}")
print("-" * 65)
print(f"{'Base':<20} {test_acc_base:<15.4f} {f1_base:<15.4f} {test_loss_base:<15.4f}")
print(f"{'Con Dropout':<20} {test_acc_dropout:<15.4f} {f1_dropout:<15.4f} {test_loss_dropout:<15.4f}")

print("\nANALISIS DE OVERFITTING:")
print(f"  Modelo Base - Gap (Train-Val): {gap_base:.4f} ({gap_base*100:.2f}%)")
print(f"  Modelo Dropout - Gap (Train-Val): {gap_dropout:.4f} ({gap_dropout*100:.2f}%)")
print(f"  Reduccion de Overfitting: {(gap_base - gap_dropout):.4f} ({(gap_base - gap_dropout)*100:.2f}%)")

print("\nCONCLUSIONES:")
if test_acc_dropout > test_acc_base:
    mejora = (test_acc_dropout - test_acc_base) * 100
    print(f"  - El modelo con Dropout supera al base en {mejora:.2f}% accuracy")
    print(f"  - Dropout mejora la generalizacion del modelo")
else:
    print(f"  - El modelo base tuvo mejor accuracy en test set")
    
if gap_dropout < gap_base:
    print(f"  - Dropout reduce exitosamente el overfitting")
else:
    print(f"  - El overfitting no se redujo significativamente")

if f1_dropout > f1_base:
    print(f"  - F1-Score mejor con Dropout (mejor balance precision/recall)")
else:
    print(f"  - F1-Score mejor sin Dropout")

print("\n" + "="*80)
print("EVALUACION COMPLETADA")
print("="*80)

print("\nArchivos generados:")
print(f"  {SAVE_DIR}/curvas_entrenamiento.png")
print(f"  {SAVE_DIR}/matrices_confusion.png")
print(f"  {SAVE_DIR}/tabla_comparativa.png")

print("\nPara hacer predicciones en nuevas imagenes:")
print("  from tensorflow import keras")
print("  import numpy as np")
print("  modelo = keras.models.load_model('modelos_entrenados/modelo_dropout.h5')")
print("  predicciones = modelo.predict(tus_imagenes)")