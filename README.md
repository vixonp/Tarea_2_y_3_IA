# T2_IA - Tarea 2 Inteligencia Artificial UDP

## Descripción

Este proyecto implementa una comparación exhaustiva de algoritmos de aprendizaje supervisado y no supervisado sobre un mismo conjunto de datos. El proyecto está dividido en dos partes principales:

### Parte 1: Clustering (30 puntos)
- Implementación y comparación de tres algoritmos de clustering:
  - K-Means
  - K-Means++
  - MeanShift
- Cada algoritmo se ejecuta con al menos 4 configuraciones distintas
- Evaluación mediante Silhouette Score
- Análisis de asignación de etiquetas basado en clusters

### Parte 2: Aprendizaje Supervisado (30 puntos)
- Entrenamiento paralelo de múltiples instancias de:
  - Regresión Logística
  - SVM (Support Vector Machine)
- Variación de hiperparámetros mediante archivo de configuración
- Evaluación periódica y descarte de peores configuraciones
- Análisis de resultados en función de hiperparámetros

## Requisitos del Dataset

- Al menos 10,000 filas
- Al menos 7 columnas utilizables
- Una columna etiqueta Y discreta con más de 2 clases

## Estructura del Proyecto

```
T2/
├── main.py                  # Script principal
├── data_loader.py           # Carga y preprocesamiento de datos
├── clustering.py            # Parte 1: Algoritmos de clustering
├── supervised_learning.py   # Parte 2: Aprendizaje supervisado
├── config.json              # Configuración de hiperparámetros
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

## Instalación

### 1. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Uso

### Ejecutar el proyecto completo

```bash
python main.py
```

### Usar un dataset personalizado

Edita `main.py` y descomenta la línea correspondiente:

```python
df = load_dataset(file_path='tu_dataset.csv')
```

Asegúrate de que tu dataset cumpla con los requisitos mencionados.

## Configuración

Los hiperparámetros para Regresión Logística y SVM se configuran en `config.json`. Puedes modificar este archivo para experimentar con diferentes configuraciones.

## Dependencias

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## Resultados

El script genera:
- Resultados de clustering con métricas de evaluación
- Comparación de configuraciones de clustering
- Evolución del entrenamiento de modelos supervisados
- Métricas de evaluación en conjunto de test
- Gráficos de resultados (si se generan correctamente)

## Notas

- El dataset por defecto es `fetch_covtype` de sklearn, que se expande automáticamente si tiene menos de 10,000 filas
- Los resultados se muestran en consola y se guardan gráficos si es posible
- El entrenamiento paralelo utiliza ThreadPoolExecutor para optimizar el tiempo de ejecución

## Autor

[Tu nombre]

## Fecha

[Fecha actual]
```

¿Quieres que actualice el README con esta versión o prefieres personalizarlo?
