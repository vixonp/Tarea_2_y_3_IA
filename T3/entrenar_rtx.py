"""
Script de entrenamiento OPTIMIZADO para RTX 3070
- Mixed Precision (FP16) para usar Tensor Cores
- Batch sizes mayores
- XLA compilation
- Monitoreo de GPU
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle
import os
from tqdm import tqdm
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN GPU RTX 3070
# ==============================================================================

print("="*80)
print("CONFIGURACIÓN GPU RTX 3070")
print("="*80)

# Detectar GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Permitir crecimiento dinámico de memoria
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detectada: {gpus[0].name}")
        print("✅ Crecimiento dinámico de memoria habilitado")
        
        # Habilitar Mixed Precision (usa Tensor Cores de RTX)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✅ Mixed Precision (FP16) habilitado - Tensor Cores activos")
        
        # Habilitar XLA
        tf.config.optimizer.set_jit(True)
        print("✅ XLA habilitado")
        
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("⚠️  GPU no detectada - usando CPU")
    print("Verifica: nvidia-smi y pip install tensorflow[and-cuda]")

# Opcional: Monitoreo GPU
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM Total: {memory_info.total / 1024**3:.2f} GB")
    pynvml.nvmlShutdown()
except:
    pass

print("="*80)

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = 'imagenes_divididas'
SAVE_DIR = './modelos_entrenados_rtx'
os.makedirs(SAVE_DIR, exist_ok=True)

# Hiperparámetros optimizados para RTX 3070
RTX_BATCH_SIZE = 128  # Aumentado de 64 (tienes 8GB VRAM)
RTX_LEARNING_RATE = 0.001  # Más alto con mixed precision
RTX_EPOCHS = 50  # Más épocas porque es rápido

# ==============================================================================
# CALLBACK PARA MONITOREO GPU
# ==============================================================================

class GPUMemoryCallback(keras.callbacks.Callback):
    """Callback personalizado para monitorear uso de GPU"""
    
    def __init__(self):
        super().__init__()
        self.has_pynvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.has_pynvml = True
        except:
            pass
    
    def on_epoch_end(self, epoch, logs=None):
        if self.has_pynvml:
            import pynvml
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            print(f"\n  GPU - Memoria: {memory_info.used / 1024**3:.2f}/{memory_info.total / 1024**3:.2f} GB "
                  f"| Uso: {utilization.gpu}% | Temp: {temperature}°C")

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def get_datasets_tf(batch_size):
    """Carga datasets con tf.data para mejor rendimiento GPU"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'training'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'validation'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        shuffle=True,
        seed=42
    )
    
    # Optimizaciones para GPU
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def build_model(architecture, kernel_size, learning_rate, use_dropout=False, dropout_rate=0.3):
    """Construye modelo CNN optimizado para RTX"""
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    # Capas convolucionales
    for num_filters in architecture:
        model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Capas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    
    # Dropout solo en FC layer
    if use_dropout:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(5, activation='softmax', dtype='float32'))  # Output siempre float32
    
    # Optimizador con learning rate adaptado a mixed precision
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Para mixed precision, wrap el optimizer
    if mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==============================================================================
# BÚSQUEDA RÁPIDA DE HIPERPARÁMETROS (menos iteraciones por velocidad GPU)
# ==============================================================================

print("\n" + "="*80)
print("BÚSQUEDA RÁPIDA DE HIPERPARÁMETROS CON RTX")
print("="*80)

# Menos opciones porque la GPU hace todo más rápido
search_params = {
    'architecture': [[32, 64], [32, 64, 128], [64, 128, 256]],
    'kernel_size': [(3, 3), (5, 5)],
    'learning_rate': [0.0001, 0.001, 0.01]
}

best_val_acc = 0
best_config = None

train_ds, val_ds = get_datasets_tf(RTX_BATCH_SIZE)

print("\nProbando arquitecturas...")
for arch in tqdm(search_params['architecture']):
    for kernel in search_params['kernel_size']:
        for lr in search_params['learning_rate']:
            
            model = build_model(arch, kernel, lr, use_dropout=False)
            
            # Solo 10 épocas para búsqueda (es rápido con GPU)
            history = model.fit(
                train_ds, 
                validation_data=val_ds, 
                epochs=10,
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = {
                    'architecture': arch,
                    'kernel_size': kernel,
                    'learning_rate': lr
                }
            
            tf.keras.backend.clear_session()

print(f"\n{'='*80}")
print("MEJOR CONFIGURACIÓN ENCONTRADA:")
print(f"  Architecture: {best_config['architecture']}")
print(f"  Kernel Size: {best_config['kernel_size']}")
print(f"  Learning Rate: {best_config['learning_rate']}")
print(f"  Val Accuracy: {best_val_acc:.4f}")
print(f"{'='*80}")

# ==============================================================================
# BÚSQUEDA DE DROPOUT
# ==============================================================================

print("\n" + "="*80)
print("BÚSQUEDA DE DROPOUT (SOLO FC LAYER)")
print("="*80)

dropout_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
dropout_results = []

train_ds, val_ds = get_datasets_tf(RTX_BATCH_SIZE)

for rate in tqdm(dropout_rates, desc="Probando dropout"):
    model = build_model(
        best_config['architecture'],
        best_config['kernel_size'],
        best_config['learning_rate'],
        use_dropout=True,
        dropout_rate=rate
    )
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=0)
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    dropout_results.append({
        'rate': rate,
        'val_acc': final_val_acc,
        'gap': overfitting_gap
    })
    
    print(f"Rate {rate}: Val={final_val_acc:.4f}, Gap={overfitting_gap:.4f}")
    tf.keras.backend.clear_session()

# Seleccionar mejor dropout
best_dropout = min(dropout_results, key=lambda x: x['gap'] if x['val_acc'] > 0.90 else 999)

print(f"\n{'='*80}")
print(f"MEJOR DROPOUT: {best_dropout['rate']}")
print(f"Val Accuracy: {best_dropout['val_acc']:.4f}")
print(f"Overfitting Gap: {best_dropout['gap']:.4f}")
print(f"{'='*80}")

# ==============================================================================
# ENTRENAMIENTO FINAL - MODELO BASE
# ==============================================================================

print("\n" + "="*80)
print("ENTRENAMIENTO FINAL - MODELO BASE")
print("="*80)

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'training'),
    target_size=(64, 64),
    batch_size=RTX_BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'validation'),
    target_size=(64, 64),
    batch_size=RTX_BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Construir modelo base
model_base = keras.Sequential(name='CNN_Base_RTX')
model_base.add(layers.Input(shape=(64, 64, 3)))
model_base.add(layers.Rescaling(1./255))

for num_filters in best_config['architecture']:
    model_base.add(layers.Conv2D(num_filters, best_config['kernel_size'], activation='relu', padding='same'))
    model_base.add(layers.MaxPooling2D((2, 2)))

model_base.add(layers.Flatten())
model_base.add(layers.Dense(128, activation='relu'))
model_base.add(layers.Dense(5, activation='softmax', dtype='float32'))

optimizer_base = optimizers.Adam(learning_rate=best_config['learning_rate'])
if mixed_precision.global_policy().name == 'mixed_float16':
    optimizer_base = mixed_precision.LossScaleOptimizer(optimizer_base)

model_base.compile(
    optimizer=optimizer_base,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_base = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, 'modelo_base_rtx.h5'), 
                   monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    GPUMemoryCallback()
]

start_time = datetime.now()
history_base = model_base.fit(
    train_generator,
    epochs=RTX_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_base,
    verbose=1
)
train_time_base = (datetime.now() - start_time).total_seconds()

print(f"\n⏱️  Tiempo de entrenamiento Base: {train_time_base/60:.2f} minutos")

# ==============================================================================
# ENTRENAMIENTO FINAL - MODELO CON DROPOUT
# ==============================================================================

print("\n" + "="*80)
print("ENTRENAMIENTO FINAL - MODELO CON DROPOUT")
print("="*80)

model_dropout = keras.Sequential(name='CNN_Dropout_RTX')
model_dropout.add(layers.Input(shape=(64, 64, 3)))
model_dropout.add(layers.Rescaling(1./255))

for num_filters in best_config['architecture']:
    model_dropout.add(layers.Conv2D(num_filters, best_config['kernel_size'], activation='relu', padding='same'))
    model_dropout.add(layers.MaxPooling2D((2, 2)))

model_dropout.add(layers.Flatten())
model_dropout.add(layers.Dense(128, activation='relu'))
model_dropout.add(layers.Dropout(best_dropout['rate']))
model_dropout.add(layers.Dense(5, activation='softmax', dtype='float32'))

optimizer_dropout = optimizers.Adam(learning_rate=best_config['learning_rate'])
if mixed_precision.global_policy().name == 'mixed_float16':
    optimizer_dropout = mixed_precision.LossScaleOptimizer(optimizer_dropout)

model_dropout.compile(
    optimizer=optimizer_dropout,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_dropout = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, 'modelo_dropout_rtx.h5'), 
                   monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1),
    GPUMemoryCallback()
]

start_time = datetime.now()
history_dropout = model_dropout.fit(
    train_generator,
    epochs=RTX_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_dropout,
    verbose=1
)
train_time_dropout = (datetime.now() - start_time).total_seconds()

print(f"\n⏱️  Tiempo de entrenamiento Dropout: {train_time_dropout/60:.2f} minutos")

# ==============================================================================
# GUARDAR RESULTADOS
# ==============================================================================

with open(os.path.join(SAVE_DIR, 'history_base_rtx.pkl'), 'wb') as f:
    pickle.dump(history_base.history, f)

with open(os.path.join(SAVE_DIR, 'history_dropout_rtx.pkl'), 'wb') as f:
    pickle.dump(history_dropout.history, f)

with open(os.path.join(SAVE_DIR, 'config_rtx.pkl'), 'wb') as f:
    pickle.dump({
        'best_config': best_config,
        'dropout_rate': best_dropout['rate'],
        'batch_size': RTX_BATCH_SIZE,
        'epochs': RTX_EPOCHS,
        'train_time_base': train_time_base,
        'train_time_dropout': train_time_dropout,
        'mixed_precision': True,
        'gpu_used': 'RTX 3070'
    }, f)

print("\n" + "="*80)
print("ENTRENAMIENTO COMPLETADO CON RTX 3070")
print("="*80)
print(f"\n📁 Archivos guardados en: {SAVE_DIR}/")
print(f"⏱️  Tiempo total: {(train_time_base + train_time_dropout)/60:.2f} minutos")
print(f"🚀 Mixed Precision: Habilitado (Tensor Cores)")
print(f"🎯 Batch Size: {RTX_BATCH_SIZE}")
print("\n✅ Usa evaluar_rtx.py para ver los resultados")
print("="*80)