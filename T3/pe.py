# Tarea 3 - Clasificación de Imágenes con CNN
# Inteligencia Artificial - Universidad Diego Portales

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuración de semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 1. CARGAR Y PREPARAR LOS DATOS
# ============================================================================

print("="*60)
print("CARGANDO DATOS...")
print("="*60)

# Definir rutas
train_dir = 'imagenes_divididas/training'
val_dir = 'imagenes_divididas/validation'
test_dir = 'imagenes_divididas/testing'

# Hiperparámetros de datos
IMG_SIZE = 64
BATCH_SIZE = 32  # Justificación: Balance entre velocidad y estabilidad del gradiente

# Crear generadores de datos con normalización
# Normalizar divide los valores de píxeles (0-255) entre 255 para obtener valores entre 0-1
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar imágenes desde directorios
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # No mezclar para mantener orden en evaluación
)

# Información del dataset
print(f"\nClases encontradas: {train_generator.class_indices}")
print(f"Número de imágenes de entrenamiento: {train_generator.samples}")
print(f"Número de imágenes de validación: {val_generator.samples}")
print(f"Número de imágenes de prueba: {test_generator.samples}")

num_classes = len(train_generator.class_indices)

# ============================================================================
# 2. VISUALIZAR ALGUNAS IMÁGENES DEL DATASET
# ============================================================================

def visualizar_muestras(generator, n_images=9):
    """Visualiza algunas imágenes del dataset"""
    x_batch, y_batch = next(generator)
    
    plt.figure(figsize=(12, 12))
    for i in range(min(n_images, len(x_batch))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_batch[i])
        clase = np.argmax(y_batch[i])
        nombre_clase = list(generator.class_indices.keys())[clase]
        plt.title(f'Clase: {nombre_clase}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('muestras_dataset.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[Imagen guardada: muestras_dataset.png]")

print("\n" + "="*60)
print("VISUALIZANDO MUESTRAS DEL DATASET...")
print("="*60)
visualizar_muestras(train_generator)

# ============================================================================
# 3. CONSTRUIR ARQUITECTURA CNN BASE (SIN DROPOUT)
# ============================================================================

print("\n" + "="*60)
print("CONSTRUYENDO ARQUITECTURA CNN BASE...")
print("="*60)

def crear_modelo_base():
    """
    Arquitectura CNN Base
    
    Justificación de hiperparámetros:
    - Kernel size 3x3: Estándar en CNNs, captura patrones locales eficientemente
    - Filtros crecientes (32->64->128): Captura características desde simples a complejas
    - MaxPooling 2x2: Reduce dimensionalidad y añade invariancia a traslación
    - Activation ReLU: Introduce no-linealidad, evita vanishing gradient
    - 3 bloques convolucionales: Suficiente para imágenes 64x64
    - Dense(128): Capa fully connected para combinar características
    - Dense(num_classes) con softmax: Clasificación multiclase
    """
    model = keras.Sequential([
        # Bloque Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque Convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten y capas densas (FCL)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_Base')
    
    return model

# Crear modelo base
model_base = crear_modelo_base()
model_base.summary()

# ============================================================================
# 4. COMPILAR Y ENTRENAR MODELO BASE
# ============================================================================

print("\n" + "="*60)
print("COMPILANDO MODELO BASE...")
print("="*60)

# Hiperparámetros de entrenamiento
LEARNING_RATE = 0.001  # Justificación: Valor estándar para Adam, buen balance
EPOCHS = 30

# Compilar modelo
# Adam: Optimizer adaptativo, combina momentum y RMSprop
# categorical_crossentropy: Para clasificación multiclase
model_base.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks para mejorar entrenamiento
callbacks_base = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'modelo_base_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*60)
print("ENTRENANDO MODELO BASE...")
print("="*60)

# Entrenar modelo
history_base = model_base.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_base,
    verbose=1
)

# ============================================================================
# 5. CONSTRUIR ARQUITECTURA CNN CON DROPOUT
# ============================================================================

print("\n" + "="*60)
print("CONSTRUYENDO ARQUITECTURA CNN CON DROPOUT...")
print("="*60)

def crear_modelo_dropout():
    """
    Arquitectura CNN con Dropout
    
    ¿Qué es Dropout?
    - Técnica de regularización que "apaga" aleatoriamente neuronas durante entrenamiento
    - Cada neurona tiene una probabilidad p de ser ignorada en cada paso
    - Previene co-adaptación de neuronas (dependencia excesiva entre ellas)
    
    ¿Cómo reduce el sobreajuste?
    - Fuerza a la red a aprender características redundantes y robustas
    - Simula un ensemble de redes, mejorando generalización
    - Evita que el modelo memorice los datos de entrenamiento
    
    Justificación de ubicación y tasas:
    - Dropout(0.25) después de MaxPooling: Regulariza características espaciales
    - Dropout(0.5) antes de capa de salida: Mayor regularización en FCL
      (las capas densas tienden a sobreajustar más)
    """
    model = keras.Sequential([
        # Bloque Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # Dropout moderado después de pooling
        
        # Bloque Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque Convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten y capas densas (FCL)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Dropout más agresivo en FCL
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_Dropout')
    
    return model

# Crear modelo con dropout
model_dropout = crear_modelo_dropout()
model_dropout.summary()

# ============================================================================
# 6. COMPILAR Y ENTRENAR MODELO CON DROPOUT
# ============================================================================

print("\n" + "="*60)
print("COMPILANDO MODELO CON DROPOUT...")
print("="*60)

model_dropout.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_dropout = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'modelo_dropout_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\n" + "="*60)
print("ENTRENANDO MODELO CON DROPOUT...")
print("="*60)

history_dropout = model_dropout.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_dropout,
    verbose=1
)

# ============================================================================
# 7. VISUALIZAR CURVAS DE ENTRENAMIENTO
# ============================================================================

def plot_training_history(history_base, history_dropout):
    """Grafica curvas de loss y accuracy para ambos modelos"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss - Modelo Base
    axes[0, 0].plot(history_base.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history_base.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Modelo Base - Loss por Época', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy - Modelo Base
    axes[0, 1].plot(history_base.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history_base.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Modelo Base - Accuracy por Época', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss - Modelo con Dropout
    axes[1, 0].plot(history_dropout.history['loss'], label='Training Loss', linewidth=2)
    axes[1, 0].plot(history_dropout.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1, 0].set_title('Modelo con Dropout - Loss por Época', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy - Modelo con Dropout
    axes[1, 1].plot(history_dropout.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1, 1].plot(history_dropout.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1, 1].set_title('Modelo con Dropout - Accuracy por Época', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[Imagen guardada: curvas_entrenamiento.png]")

print("\n" + "="*60)
print("VISUALIZANDO CURVAS DE ENTRENAMIENTO...")
print("="*60)
plot_training_history(history_base, history_dropout)

# ============================================================================
# 8. EVALUACIÓN EN CONJUNTO DE PRUEBA
# ============================================================================

print("\n" + "="*60)
print("EVALUANDO MODELOS EN CONJUNTO DE PRUEBA...")
print("="*60)

# Evaluar modelo base
test_loss_base, test_acc_base = model_base.evaluate(test_generator, verbose=0)
print(f"\nModelo Base:")
print(f"  Test Loss: {test_loss_base:.4f}")
print(f"  Test Accuracy: {test_acc_base:.4f}")

# Evaluar modelo con dropout
test_loss_dropout, test_acc_dropout = model_dropout.evaluate(test_generator, verbose=0)
print(f"\nModelo con Dropout:")
print(f"  Test Loss: {test_loss_dropout:.4f}")
print(f"  Test Accuracy: {test_acc_dropout:.4f}")

# Predicciones para métricas adicionales
y_pred_base = model_base.predict(test_generator, verbose=0)
y_pred_dropout = model_dropout.predict(test_generator, verbose=0)

# Convertir predicciones a clases
y_pred_classes_base = np.argmax(y_pred_base, axis=1)
y_pred_classes_dropout = np.argmax(y_pred_dropout, axis=1)
y_true = test_generator.classes

# Nombres de clases
class_names = list(test_generator.class_indices.keys())

# ============================================================================
# 9. MÉTRICAS ADICIONALES
# ============================================================================

print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN - MODELO BASE")
print("="*60)
print(classification_report(y_true, y_pred_classes_base, target_names=class_names))

print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN - MODELO CON DROPOUT")
print("="*60)
print(classification_report(y_true, y_pred_classes_dropout, target_names=class_names))

# ============================================================================
# 10. MATRICES DE CONFUSIÓN
# ============================================================================

def plot_confusion_matrices(y_true, y_pred_base, y_pred_dropout, class_names):
    """Grafica matrices de confusión para ambos modelos"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Matriz de confusión - Modelo Base
    cm_base = confusion_matrix(y_true, y_pred_base)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Matriz de Confusión - Modelo Base', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Etiqueta Verdadera')
    axes[0].set_xlabel('Etiqueta Predicha')
    
    # Matriz de confusión - Modelo con Dropout
    cm_dropout = confusion_matrix(y_true, y_pred_dropout)
    sns.heatmap(cm_dropout, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Matriz de Confusión - Modelo con Dropout', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Etiqueta Verdadera')
    axes[1].set_xlabel('Etiqueta Predicha')
    
    plt.tight_layout()
    plt.savefig('matrices_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[Imagen guardada: matrices_confusion.png]")

print("\n" + "="*60)
print("GENERANDO MATRICES DE CONFUSIÓN...")
print("="*60)
plot_confusion_matrices(y_true, y_pred_classes_base, y_pred_classes_dropout, class_names)

# ============================================================================
# 11. RESUMEN COMPARATIVO
# ============================================================================

print("\n" + "="*60)
print("RESUMEN COMPARATIVO FINAL")
print("="*60)

print("\nMÉTRICAS EN CONJUNTO DE PRUEBA:")
print(f"{'Modelo':<20} {'Test Loss':<15} {'Test Accuracy':<15}")
print("-" * 50)
print(f"{'Base':<20} {test_loss_base:<15.4f} {test_acc_base:<15.4f}")
print(f"{'Con Dropout':<20} {test_loss_dropout:<15.4f} {test_acc_dropout:<15.4f}")

print("\n" + "="*60)
print("¡ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS!")
print("="*60)
print("\nArchivos generados:")
print("  - muestras_dataset.png")
print("  - curvas_entrenamiento.png")
print("  - matrices_confusion.png")
print("  - modelo_base_best.h5")
print("  - modelo_dropout_best.h5")