import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = 'imagenes_divididas'
SAVE_DIR = './modelos_entrenados'
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================================================
# BUSQUEDA DE HIPERPARAMETROS
# ==============================================================================

def get_datasets_tf(batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'training'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        verbose=0
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(BASE_DIR, 'validation'),
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int',
        verbose=0
    )
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def build_model_search(architecture, kernel_size, learning_rate):
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    for num_filters in architecture:
        model.add(layers.Conv2D(num_filters, kernel_size, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("="*60)
print("BUSQUEDA DE HIPERPARAMETROS")
print("="*60)

hyperparams = {
    'batch_size': {'values': [32, 64, 128], 'accuracy': [], 'best_index': 0},
    'epochs': {'values': [10, 20, 30], 'accuracy': [], 'best_index': 0},
    'learning_rate': {'values': [0.0001, 0.001, 0.01], 'accuracy': [], 'best_index': 0},
    'kernel_size': {'values': [(3, 3), (5, 5)], 'accuracy': [], 'best_index': 0},
    'architecture': {'values': [[32, 64], [32, 64, 128], [64, 128, 256]], 'accuracy': [], 'best_index': 0}
}

print("\nProbando Batch Size...")
for i, val in enumerate(tqdm(hyperparams['batch_size']['values'])):
    train_ds, val_ds = get_datasets_tf(val)
    model = build_model_search(
        architecture=hyperparams['architecture']['values'][hyperparams['architecture']['best_index']],
        kernel_size=hyperparams['kernel_size']['values'][hyperparams['kernel_size']['best_index']],
        learning_rate=hyperparams['learning_rate']['values'][hyperparams['learning_rate']['best_index']]
    )
    history = model.fit(train_ds, validation_data=val_ds, 
                       epochs=hyperparams['epochs']['values'][hyperparams['epochs']['best_index']], 
                       verbose=0)
    accuracy = max(history.history['val_accuracy'])
    hyperparams['batch_size']['accuracy'].append(accuracy)
    if accuracy > hyperparams['batch_size']['accuracy'][hyperparams['batch_size']['best_index']]:
        hyperparams['batch_size']['best_index'] = i
    tf.keras.backend.clear_session()

print("Probando Epochs...")
for i, val in enumerate(tqdm(hyperparams['epochs']['values'])):
    train_ds, val_ds = get_datasets_tf(hyperparams['batch_size']['values'][hyperparams['batch_size']['best_index']])
    model = build_model_search(
        architecture=hyperparams['architecture']['values'][hyperparams['architecture']['best_index']],
        kernel_size=hyperparams['kernel_size']['values'][hyperparams['kernel_size']['best_index']],
        learning_rate=hyperparams['learning_rate']['values'][hyperparams['learning_rate']['best_index']]
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=val, verbose=0)
    accuracy = max(history.history['val_accuracy'])
    hyperparams['epochs']['accuracy'].append(accuracy)
    if accuracy > hyperparams['epochs']['accuracy'][hyperparams['epochs']['best_index']]:
        hyperparams['epochs']['best_index'] = i
    tf.keras.backend.clear_session()

print("Probando Learning Rate...")
for i, val in enumerate(tqdm(hyperparams['learning_rate']['values'])):
    train_ds, val_ds = get_datasets_tf(hyperparams['batch_size']['values'][hyperparams['batch_size']['best_index']])
    model = build_model_search(
        architecture=hyperparams['architecture']['values'][hyperparams['architecture']['best_index']],
        kernel_size=hyperparams['kernel_size']['values'][hyperparams['kernel_size']['best_index']],
        learning_rate=val
    )
    history = model.fit(train_ds, validation_data=val_ds, 
                       epochs=hyperparams['epochs']['values'][hyperparams['epochs']['best_index']], 
                       verbose=0)
    accuracy = max(history.history['val_accuracy'])
    hyperparams['learning_rate']['accuracy'].append(accuracy)
    if accuracy > hyperparams['learning_rate']['accuracy'][hyperparams['learning_rate']['best_index']]:
        hyperparams['learning_rate']['best_index'] = i
    tf.keras.backend.clear_session()

print("Probando Kernel Size...")
for i, val in enumerate(tqdm(hyperparams['kernel_size']['values'])):
    train_ds, val_ds = get_datasets_tf(hyperparams['batch_size']['values'][hyperparams['batch_size']['best_index']])
    model = build_model_search(
        architecture=hyperparams['architecture']['values'][hyperparams['architecture']['best_index']],
        kernel_size=val,
        learning_rate=hyperparams['learning_rate']['values'][hyperparams['learning_rate']['best_index']]
    )
    history = model.fit(train_ds, validation_data=val_ds, 
                       epochs=hyperparams['epochs']['values'][hyperparams['epochs']['best_index']], 
                       verbose=0)
    accuracy = max(history.history['val_accuracy'])
    hyperparams['kernel_size']['accuracy'].append(accuracy)
    if accuracy > hyperparams['kernel_size']['accuracy'][hyperparams['kernel_size']['best_index']]:
        hyperparams['kernel_size']['best_index'] = i
    tf.keras.backend.clear_session()

print("Probando Arquitecturas...")
for i, val in enumerate(tqdm(hyperparams['architecture']['values'])):
    train_ds, val_ds = get_datasets_tf(hyperparams['batch_size']['values'][hyperparams['batch_size']['best_index']])
    model = build_model_search(
        architecture=val,
        kernel_size=hyperparams['kernel_size']['values'][hyperparams['kernel_size']['best_index']],
        learning_rate=hyperparams['learning_rate']['values'][hyperparams['learning_rate']['best_index']]
    )
    history = model.fit(train_ds, validation_data=val_ds, 
                       epochs=hyperparams['epochs']['values'][hyperparams['epochs']['best_index']], 
                       verbose=0)
    accuracy = max(history.history['val_accuracy'])
    hyperparams['architecture']['accuracy'].append(accuracy)
    if accuracy > hyperparams['architecture']['accuracy'][hyperparams['architecture']['best_index']]:
        hyperparams['architecture']['best_index'] = i
    tf.keras.backend.clear_session()

BEST_BATCH_SIZE = hyperparams['batch_size']['values'][hyperparams['batch_size']['best_index']]
BEST_EPOCHS = hyperparams['epochs']['values'][hyperparams['epochs']['best_index']]
BEST_LR = hyperparams['learning_rate']['values'][hyperparams['learning_rate']['best_index']]
BEST_KERNEL = hyperparams['kernel_size']['values'][hyperparams['kernel_size']['best_index']]
BEST_ARCH = hyperparams['architecture']['values'][hyperparams['architecture']['best_index']]

print("\n" + "="*60)
print("MEJORES HIPERPARAMETROS:")
print(f"  Batch Size: {BEST_BATCH_SIZE}")
print(f"  Epochs: {BEST_EPOCHS}")
print(f"  Learning Rate: {BEST_LR}")
print(f"  Kernel Size: {BEST_KERNEL}")
print(f"  Architecture: {BEST_ARCH}")
print("="*60)

# Guardar hiperparametros
with open(os.path.join(SAVE_DIR, 'best_hyperparams.pkl'), 'wb') as f:
    pickle.dump({
        'batch_size': BEST_BATCH_SIZE,
        'epochs': BEST_EPOCHS,
        'learning_rate': BEST_LR,
        'kernel_size': BEST_KERNEL,
        'architecture': BEST_ARCH
    }, f)

# ==============================================================================
# BUSQUEDA DE TASA DE DROPOUT
# ==============================================================================

print("\n" + "="*60)
print("BUSQUEDA DE TASA DE DROPOUT")
print("="*60)

dropout_rates = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
dropout_accuracies = []

train_ds, val_ds = get_datasets_tf(BEST_BATCH_SIZE)

for rate in tqdm(dropout_rates):
    model = models.Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Rescaling(1./255))
    
    for num_filters in BEST_ARCH:
        model.add(layers.Conv2D(num_filters, BEST_KERNEL, activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(rate))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate))
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=BEST_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=BEST_EPOCHS, verbose=0)
    dropout_accuracies.append(max(history.history['val_accuracy']))
    tf.keras.backend.clear_session()

best_dropout_idx = np.argmax(dropout_accuracies)
BEST_DROPOUT = dropout_rates[best_dropout_idx]

print(f"\nMejor tasa de Dropout: {BEST_DROPOUT}")

# ==============================================================================
# ENTRENAMIENTO FINAL DE MODELOS
# ==============================================================================

print("\n" + "="*60)
print("ENTRENANDO MODELO BASE (SIN DROPOUT)")
print("="*60)

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'training'),
    target_size=(64, 64),
    batch_size=BEST_BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_gen.flow_from_directory(
    os.path.join(BASE_DIR, 'validation'),
    target_size=(64, 64),
    batch_size=BEST_BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

model_base = keras.Sequential(name='CNN_Base')
model_base.add(layers.Input(shape=(64, 64, 3)))

for num_filters in BEST_ARCH:
    model_base.add(layers.Conv2D(num_filters, BEST_KERNEL, activation='relu', padding='same'))
    model_base.add(layers.MaxPooling2D((2, 2)))

model_base.add(layers.Flatten())
model_base.add(layers.Dense(128, activation='relu'))
model_base.add(layers.Dense(5, activation='softmax'))

model_base.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BEST_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_base = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, 'modelo_base.h5'), monitor='val_accuracy', 
                   save_best_only=True, verbose=1)
]

history_base = model_base.fit(
    train_generator,
    epochs=BEST_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_base,
    verbose=1
)

with open(os.path.join(SAVE_DIR, 'history_base.pkl'), 'wb') as f:
    pickle.dump(history_base.history, f)

print("\n" + "="*60)
print("ENTRENANDO MODELO CON DROPOUT")
print("="*60)

model_dropout = keras.Sequential(name='CNN_Dropout')
model_dropout.add(layers.Input(shape=(64, 64, 3)))

for num_filters in BEST_ARCH:
    model_dropout.add(layers.Conv2D(num_filters, BEST_KERNEL, activation='relu', padding='same'))
    model_dropout.add(layers.MaxPooling2D((2, 2)))
    model_dropout.add(layers.Dropout(BEST_DROPOUT))

model_dropout.add(layers.Flatten())
model_dropout.add(layers.Dense(128, activation='relu'))
model_dropout.add(layers.Dropout(BEST_DROPOUT))
model_dropout.add(layers.Dense(5, activation='softmax'))

model_dropout.compile(
    optimizer=keras.optimizers.Adam(learning_rate=BEST_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_dropout = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(os.path.join(SAVE_DIR, 'modelo_dropout.h5'), monitor='val_accuracy', 
                   save_best_only=True, verbose=1)
]

history_dropout = model_dropout.fit(
    train_generator,
    epochs=BEST_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_dropout,
    verbose=1
)

with open(os.path.join(SAVE_DIR, 'history_dropout.pkl'), 'wb') as f:
    pickle.dump(history_dropout.history, f)

# Guardar información del dropout usado
with open(os.path.join(SAVE_DIR, 'dropout_info.pkl'), 'wb') as f:
    pickle.dump({'dropout_rate': BEST_DROPOUT}, f)

print("\n" + "="*80)
print("ENTRENAMIENTO COMPLETADO")
print("="*80)
print(f"\nArchivos guardados en: {SAVE_DIR}/")
print("  - modelo_base.h5")
print("  - modelo_dropout.h5")
print("  - history_base.pkl")
print("  - history_dropout.pkl")
print("  - best_hyperparams.pkl")
print("  - dropout_info.pkl")
print("\nEjecuta evaluar.py para ver resultados y metricas.")
print("="*80)