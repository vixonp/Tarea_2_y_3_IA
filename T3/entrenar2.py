import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pickle
import os
from tqdm import tqdm
from datetime import datetime

# configurar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detectada correctamente: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("GPU no detectada - usando CPU")

print("="*80)

# seeds
np.random.seed(77)
tf.random.set_seed(77)

# directorios
BASE_DIR = './imagenes_divididas'
SAVE_DIR = './modelos_entrenados'
os.makedirs(SAVE_DIR, exist_ok=True)

# === funciones ===

def get_datasets(batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'training'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        shuffle=True,
        seed=42,
        verbose=0
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'validation'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        shuffle=False,
        seed=42,
        verbose=0
    )
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

# === construir CNN ===

def build_model(architecture, kernel_size, dropout_rate=None):
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    # capas convolucionales
    for num_filters in architecture:
        model.add(layers.Conv2D(num_filters, kernel_size, 
                               activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # capas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    
    # dropout opcional
    if dropout_rate is not None:
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(5, activation='softmax'))
    
    return model 

# === busqueda de hiperparametros ===

print("\n" + "="*80)
print("BUSQUEDA DE HIPERPARAMETROS CON VALIDACION")
print("="*80)

hyperparams = {
    'batch_size': [64, 128],
    'epochs': [15, 20],
    'learning_rate': [0.0001, 0.001],
    'kernel_size': [(3, 3), (5, 5)],
    'architecture': [[32, 64], [32, 64, 128]]
}

print("\nHiperparametros a explorar:")
for param, values in hyperparams.items():
    print(f"  {param}: {values}")

best_params = {
    'batch_size': 128,
    'epochs': 15,
    'learning_rate': 0.0001,
    'kernel_size': (5, 5),
    'architecture': [32, 64, 128]
}

# 1. BATCH SIZE
print("\nProbando Batch Size...")
batch_results = []
for bs in tqdm(hyperparams['batch_size']):
    train_ds, val_ds = get_datasets(bs)
    model = build_model(best_params['architecture'], best_params['kernel_size'])
    model.compile(optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
    val_acc = max(history.history['val_accuracy'])
    batch_results.append({'value': bs, 'val_acc': val_acc})
    tqdm.write(f"  Batch Size {bs}: Val Acc = {val_acc:.4f}")
    tf.keras.backend.clear_session()

best_params['batch_size'] = max(batch_results, key=lambda x: x['val_acc'])['value']
print(f"Mejor Batch Size: {best_params['batch_size']}")

# 2. ARCHITECTURE
print("\nProbando Arquitecturas...")
arch_results = []
for arch in tqdm(hyperparams['architecture']):
    train_ds, val_ds = get_datasets(best_params['batch_size'])
    model = build_model(arch, best_params['kernel_size'])
    model.compile(optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
    val_acc = max(history.history['val_accuracy'])
    arch_results.append({'value': arch, 'val_acc': val_acc})
    tqdm.write(f"  Architecture {arch}: Val Acc = {val_acc:.4f}")
    tf.keras.backend.clear_session()

best_params['architecture'] = max(arch_results, key=lambda x: x['val_acc'])['value']
print(f"Mejor Arquitectura: {best_params['architecture']}")

# 3. KERNEL SIZE
print("\nProbando Kernel Size...")
kernel_results = []
for ks in tqdm(hyperparams['kernel_size']):
    train_ds, val_ds = get_datasets(best_params['batch_size'])
    model = build_model(best_params['architecture'], ks)
    model.compile(optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
    val_acc = max(history.history['val_accuracy'])
    kernel_results.append({'value': ks, 'val_acc': val_acc})
    tqdm.write(f"  Kernel {ks}: Val Acc = {val_acc:.4f}")
    tf.keras.backend.clear_session()

best_params['kernel_size'] = max(kernel_results, key=lambda x: x['val_acc'])['value']
print(f"Mejor Kernel Size: {best_params['kernel_size']}")

# 4. LEARNING RATE
print("\nProbando Learning Rate...")
lr_results = []
for lr in tqdm(hyperparams['learning_rate']):
    train_ds, val_ds = get_datasets(best_params['batch_size'])
    model = build_model(best_params['architecture'], best_params['kernel_size'])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=15, verbose=0)
    val_acc = max(history.history['val_accuracy'])
    lr_results.append({'value': lr, 'val_acc': val_acc})
    tqdm.write(f"  Learning Rate {lr}: Val Acc = {val_acc:.4f}")
    tf.keras.backend.clear_session()

best_params['learning_rate'] = max(lr_results, key=lambda x: x['val_acc'])['value']
print(f"Mejor Learning Rate: {best_params['learning_rate']}")

# 5. EPOCHS
print("\nProbando Epochs...")
epoch_results = []
for ep in tqdm(hyperparams['epochs']):
    train_ds, val_ds = get_datasets(best_params['batch_size'])
    model = build_model(best_params['architecture'], best_params['kernel_size'])
    model.compile(optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=ep, verbose=0)
    val_acc = max(history.history['val_accuracy'])
    epoch_results.append({'value': ep, 'val_acc': val_acc})
    tqdm.write(f"  Epochs {ep}: Val Acc = {val_acc:.4f}")
    tf.keras.backend.clear_session()

best_params['epochs'] = max(epoch_results, key=lambda x: x['val_acc'])['value']
print(f"Mejor Epochs: {best_params['epochs']}")

print("\n" + "="*80)
print("HIPERPARAMETROS OPTIMOS:")
print("="*80)
for param, value in best_params.items():
    print(f"  {param}: {value}")
print("="*80)

# === busqueda de tasa de dropout ===

print("\n" + "="*80)
print("BUSQUEDA DE TASA DE DROPOUT")
print("="*80)

dropout_val = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
dropout_accuracy = []

print("\nIniciando prueba de dropout")
print("-"*80)

train_ds, val_ds = get_datasets(best_params['batch_size'])

for i, val in enumerate(tqdm(dropout_val, desc="Probando dropout")):
    tqdm.write(f"Probando porcentaje de dropout = {val}")
    
    model = build_model(best_params['architecture'], best_params['kernel_size'], dropout_rate=val)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=0
    )
    
    max_val_acc = max(history.history['val_accuracy'])
    dropout_accuracy.append(max_val_acc)
    
    tqdm.write(f"  Resultado: Val Accuracy = {max_val_acc:.4f}")
    
    tf.keras.backend.clear_session()
    del model

print("\n" + "="*80)
print("RESULTADOS FINALES - BUSQUEDA DE DROPOUT")
print("="*80)
for i, val in enumerate(dropout_val):
    print(f"{i+1}. Dropout = {val} -> Val Accuracy = {dropout_accuracy[i]:.4f}")

best_idx = dropout_accuracy.index(max(dropout_accuracy))
BEST_DROPOUT = dropout_val[best_idx]

print("\n" + "="*80)
print(f"Mejor tasa de dropout: {BEST_DROPOUT}")
print(f"Val Accuracy: {dropout_accuracy[best_idx]:.4f}")
print("="*80)

# === entrenamiento final ===

print("\n" + "="*80)
print("ENTRENAMIENTO FINAL")
print("="*80)

train_ds, val_ds = get_datasets(best_params['batch_size'])

# modelo sin dropout
print("\n--- Entrenando Modelo SIN Dropout ---")
model = build_model(best_params['architecture'], best_params['kernel_size'], dropout_rate=None)
model.compile(
    optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = datetime.now()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=best_params['epochs'],
    verbose=1
)
time_base = (datetime.now() - start).total_seconds()

model.save(os.path.join(SAVE_DIR, 'modelo_no_dropout.keras'))

with open(os.path.join(SAVE_DIR, 'history_no_dropout.pkl'), 'wb') as f:
    pickle.dump(history.history, f)

print(f"Modelo sin dropout guardado ({time_base/60:.2f} min)")

tf.keras.backend.clear_session()
del model

# modelo con dropout
print(f"\n--- Entrenando Modelo CON Dropout (rate={BEST_DROPOUT}) ---")
model_dropout = build_model(best_params['architecture'], best_params['kernel_size'], dropout_rate=BEST_DROPOUT)
model_dropout.compile(
    optimizer=optimizers.Adam(learning_rate=best_params['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

start = datetime.now()
history_dropout = model_dropout.fit(
    train_ds,
    validation_data=val_ds,
    epochs=best_params['epochs'],
    verbose=1
)
time_dropout = (datetime.now() - start).total_seconds()

model_dropout.save(os.path.join(SAVE_DIR, 'modelo_dropout.keras'))

with open(os.path.join(SAVE_DIR, 'history_dropout.pkl'), 'wb') as f:
    pickle.dump(history_dropout.history, f)

print(f"Modelo con dropout guardado ({time_dropout/60:.2f} min)")
print(f"\nEntrenamiento finalizado! Modelos e historiales guardados en {SAVE_DIR}")

# === resumen ===

print("\n" + "="*80)
print("ENTRENAMIENTO COMPLETADO")
print("="*80)

print("\nArchivos guardados en:", SAVE_DIR)
print("  - modelo_no_dropout.keras")
print("  - modelo_dropout.keras")
print("  - history_no_dropout.pkl")
print("  - history_dropout.pkl")

print("\nMetricas finales:")
print(f"\nModelo SIN Dropout:")
print(f"  Train Acc: {history.history['accuracy'][-1]:.4f}")
print(f"  Val Acc:   {history.history['val_accuracy'][-1]:.4f}")

print(f"\nModelo CON Dropout:")
print(f"  Train Acc: {history_dropout.history['accuracy'][-1]:.4f}")
print(f"  Val Acc:   {history_dropout.history['val_accuracy'][-1]:.4f}")

print(f"\nTiempos de entrenamiento:")
print(f"  Sin Dropout: {time_base/60:.2f} minutos")
print(f"  Con Dropout: {time_dropout/60:.2f} minutos")
print(f"  Total:       {(time_base + time_dropout)/60:.2f} minutos")

print("\n" + "="*80)
print("Siguiente paso: Evaluar en conjunto de prueba (evaluacion.py)")
print("="*80)