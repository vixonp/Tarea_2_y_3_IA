"""
Script para verificar y probar configuración de GPU RTX 3070
Ejecutar ANTES del entrenamiento para confirmar que todo funciona
"""

import sys
import tensorflow as tf
import numpy as np
from datetime import datetime

print("="*80)
print("VERIFICACION DE GPU - RTX 3070")
print("="*80)

# ==============================================================================
# 1. INFORMACIÓN DEL SISTEMA
# ==============================================================================
print("\n[1] VERSIONES DE SOFTWARE")
print("-"*80)
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")

# ==============================================================================
# 2. DETECCIÓN DE GPU
# ==============================================================================
print("\n[2] DETECCIÓN DE DISPOSITIVOS")
print("-"*80)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"CPUs detectadas: {len(cpus)}")
print(f"GPUs detectadas: {len(gpus)}")

if gpus:
    print("\n✅ GPU(s) ENCONTRADA(S):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        print(f"  Tipo: {gpu.device_type}")
    
    # Información detallada con cuda
    if tf.test.is_built_with_cuda():
        print(f"\n✅ TensorFlow compilado con CUDA: SÍ")
        print(f"✅ GPU disponible para TensorFlow: {tf.test.is_gpu_available(cuda_only=True)}")
    else:
        print(f"\n⚠️  TensorFlow NO compilado con CUDA")
else:
    print("\n❌ NO SE DETECTÓ GPU")
    print("Verifica:")
    print("  1. Drivers NVIDIA instalados (nvidia-smi)")
    print("  2. CUDA Toolkit instalado")
    print("  3. TensorFlow con GPU: pip install tensorflow[and-cuda]")
    sys.exit(1)

# ==============================================================================
# 3. CONFIGURACIÓN DE GPU
# ==============================================================================
print("\n[3] CONFIGURANDO GPU PARA MÁXIMO RENDIMIENTO")
print("-"*80)

try:
    # Permitir crecimiento dinámico de memoria
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ Crecimiento dinámico de memoria habilitado")
except RuntimeError as e:
    print(f"⚠️  Error configurando memoria: {e}")

# Habilitar Mixed Precision (usa Tensor Cores)
from tensorflow.keras import mixed_precision
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✅ Mixed Precision (FP16) habilitado - usando Tensor Cores")
except Exception as e:
    print(f"⚠️  Error habilitando mixed precision: {e}")

# Habilitar XLA (compilación optimizada)
try:
    tf.config.optimizer.set_jit(True)
    print("✅ XLA (Accelerated Linear Algebra) habilitado")
except Exception as e:
    print(f"⚠️  Error habilitando XLA: {e}")

# ==============================================================================
# 4. INFORMACIÓN DETALLADA DE GPU
# ==============================================================================
print("\n[4] INFORMACIÓN DETALLADA DE GPU")
print("-"*80)

try:
    import pynvml
    pynvml.nvmlInit()
    
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Información básica
    name = pynvml.nvmlDeviceGetName(handle)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
    
    print(f"Nombre: {name}")
    print(f"Memoria Total: {memory_info.total / 1024**3:.2f} GB")
    print(f"Memoria Usada: {memory_info.used / 1024**3:.2f} GB")
    print(f"Memoria Libre: {memory_info.free / 1024**3:.2f} GB")
    print(f"Utilización GPU: {utilization.gpu}%")
    print(f"Utilización Memoria: {utilization.memory}%")
    print(f"Temperatura: {temperature}°C")
    print(f"Consumo de Energía: {power:.2f} W")
    
    pynvml.nvmlShutdown()
    
except ImportError:
    print("⚠️  pynvml no instalado. Instala con: pip install pynvml")
    print("Usando información básica de TensorFlow...")
    
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        details = tf.config.experimental.get_device_details(device)
        print(f"Device: {device}")
        print(f"Details: {details}")

# ==============================================================================
# 5. BENCHMARK DE RENDIMIENTO
# ==============================================================================
print("\n[5] BENCHMARK DE RENDIMIENTO")
print("-"*80)

def benchmark_matmul(device_name, size=5000, iterations=5):
    """Benchmark de multiplicación de matrices"""
    with tf.device(device_name):
        # Crear matrices aleatorias
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # Warmup
        _ = tf.matmul(a, b)
        
        # Benchmark
        start = datetime.now()
        for _ in range(iterations):
            c = tf.matmul(a, b)
        # Forzar ejecución
        _ = c.numpy()
        elapsed = (datetime.now() - start).total_seconds()
        
        avg_time = elapsed / iterations
        gflops = (2 * size**3) / (avg_time * 1e9)
        
        return avg_time, gflops

print("\nProbando multiplicación de matrices 5000x5000...")

# CPU Benchmark
print("\n  CPU:")
try:
    cpu_time, cpu_gflops = benchmark_matmul('/CPU:0')
    print(f"    Tiempo promedio: {cpu_time:.4f} segundos")
    print(f"    Rendimiento: {cpu_gflops:.2f} GFLOPS")
except Exception as e:
    print(f"    Error: {e}")

# GPU Benchmark
print("\n  GPU (RTX 3070):")
try:
    gpu_time, gpu_gflops = benchmark_matmul('/GPU:0')
    print(f"    Tiempo promedio: {gpu_time:.4f} segundos")
    print(f"    Rendimiento: {gpu_gflops:.2f} GFLOPS")
    
    if cpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\n  🚀 Speedup GPU vs CPU: {speedup:.2f}x más rápido")
except Exception as e:
    print(f"    Error: {e}")

# ==============================================================================
# 6. BENCHMARK CNN SIMPLIFICADO
# ==============================================================================
print("\n[6] BENCHMARK CNN (SIMULACIÓN DE TU TAREA)")
print("-"*80)

def benchmark_cnn(device_name, batch_size=128, iterations=3):
    """Benchmark con una CNN similar a tu tarea"""
    with tf.device(device_name):
        # Modelo simple similar al tuyo
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Datos dummy
        x = tf.random.normal([batch_size * 10, 64, 64, 3])
        y = tf.random.uniform([batch_size * 10], maxval=5, dtype=tf.int32)
        
        # Warmup
        model.fit(x[:batch_size], y[:batch_size], epochs=1, verbose=0)
        
        # Benchmark
        start = datetime.now()
        history = model.fit(x, y, batch_size=batch_size, epochs=iterations, verbose=0)
        elapsed = (datetime.now() - start).total_seconds()
        
        return elapsed / iterations

print(f"\nEntrenando CNN (batch_size={128}, 3 épocas)...")

print("\n  CPU:")
try:
    cpu_epoch_time = benchmark_cnn('/CPU:0', batch_size=64)  # Batch más pequeño para CPU
    print(f"    Tiempo por época: {cpu_epoch_time:.2f} segundos")
except Exception as e:
    print(f"    Error: {e}")

print("\n  GPU (RTX 3070):")
try:
    gpu_epoch_time = benchmark_cnn('/GPU:0', batch_size=128)
    print(f"    Tiempo por época: {gpu_epoch_time:.2f} segundos")
    
    if 'cpu_epoch_time' in locals():
        speedup = cpu_epoch_time / gpu_epoch_time
        print(f"\n  🚀 Speedup GPU vs CPU: {speedup:.2f}x más rápido")
        print(f"\n  📊 Para tu tarea (30 épocas):")
        print(f"     CPU: ~{cpu_epoch_time * 30 / 60:.1f} minutos")
        print(f"     GPU: ~{gpu_epoch_time * 30 / 60:.1f} minutos")
except Exception as e:
    print(f"    Error: {e}")

# ==============================================================================
# 7. RECOMENDACIONES
# ==============================================================================
print("\n" + "="*80)
print("RECOMENDACIONES PARA TU ENTRENAMIENTO")
print("="*80)

if gpus:
    print("\n✅ GPU configurada correctamente. Recomendaciones:")
    print("  1. Usa batch_size=128 o 256 (tienes 8GB VRAM)")
    print("  2. Mixed precision ya habilitado (2x más rápido)")
    print("  3. XLA habilitado (optimización automática)")
    print("  4. Monitorea temperatura con: nvidia-smi -l 1")
    print("\n  Ejemplo en tu código:")
    print("     BEST_BATCH_SIZE = 128  # Aumenta de 64")
    print("     # Mixed precision ya configurado globalmente")
else:
    print("\n❌ GPU no detectada. Soluciones:")
    print("  1. Ejecuta: nvidia-smi")
    print("  2. Instala drivers NVIDIA más recientes")
    print("  3. Reinstala TensorFlow GPU: pip install tensorflow[and-cuda]")

print("\n" + "="*80)
print("VERIFICACIÓN COMPLETA")
print("="*80)
print("\n✅ Todo listo para entrenar con RTX 3070!\n")