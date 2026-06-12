"""Asegura que la raíz del proyecto esté en sys.path para que `from src import ...`
y `from config import ...` funcionen al ejecutar pytest desde cualquier sitio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
