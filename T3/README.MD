# Si no existe crear entorno virtual
python3 -m venv .venv

# Desplegar entorno virtual
source .venv/bin/activate 

# Desplegar dependencias para mac
pip install -r requirements.txt

# Desplegar dependencias para windows
pip install -r requirementsWindows.txt