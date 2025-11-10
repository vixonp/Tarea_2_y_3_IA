"""
Script principal que ejecuta ambas actividades de la Tarea 2.
Para ejecutar actividades individuales, usa:
- python actividad1_clustering.py
- python actividad2_supervised.py
"""

import sys

def main():
    print("=" * 80)
    print("TAREA 2: COMPARACIÓN DE ALGORITMOS DE APRENDIZAJE")
    print("=" * 80)
    print("\nEste script ejecuta ambas actividades.")
    print("Para ejecutar actividades individuales, usa:")
    print("  - python actividad1_clustering.py")
    print("  - python actividad2_supervised.py")
    print("\n" + "=" * 80)
    
    # Ejecutar Actividad 1
    print("\n>>> EJECUTANDO ACTIVIDAD 1...")
    print("=" * 80)
    from actividad1_clustering import main as actividad1_main
    actividad1_main()
    
    # Ejecutar Actividad 2
    print("\n\n>>> EJECUTANDO ACTIVIDAD 2...")
    print("=" * 80)
    from actividad2_supervised import main as actividad2_main
    actividad2_main()
    
    print("\n" + "=" * 80)
    print("TODAS LAS ACTIVIDADES COMPLETADAS")
    print("=" * 80)

if __name__ == "__main__":
    main()