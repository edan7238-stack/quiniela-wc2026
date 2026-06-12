#!/bin/bash
# ============================================================
# Lanzador del dashboard del Modelo Probabilístico WC 2026.
# Mac/Linux: abre una terminal, arrastra este archivo y pulsa Enter.
# Para cerrarlo: pulsa Ctrl+C en la terminal.
# ============================================================

# Moverse a la carpeta donde está este script (raíz del proyecto)
cd "$(dirname "$0")"

# -----------------------------------------------------------
# 1. Verificar que Python 3.11+ está disponible
# -----------------------------------------------------------
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        MAJOR=$("$cmd" -c "import sys; print(sys.version_info[0])" 2>/dev/null)
        MINOR=$("$cmd" -c "import sys; print(sys.version_info[1])" 2>/dev/null)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo ""
    echo "❌  No se encontró Python 3.11 o superior."
    echo "    Descárgalo en: https://www.python.org/downloads/"
    echo ""
    read -p "Pulsa Enter para salir..."
    exit 1
fi

echo ""
echo "✅  Usando: $($PYTHON --version)"

# -----------------------------------------------------------
# 2. Suprimir el aviso de correo de Streamlit
# -----------------------------------------------------------
STREAMLIT_CREDS="$HOME/.streamlit/credentials.toml"
if [ ! -f "$STREAMLIT_CREDS" ]; then
    mkdir -p "$HOME/.streamlit"
    printf '[general]\nemail = ""\n' > "$STREAMLIT_CREDS"
fi

# -----------------------------------------------------------
# 3. Instalar / actualizar dependencias
# -----------------------------------------------------------
echo ""
echo "📦  Instalando dependencias (solo la primera vez tarda un poco)..."
$PYTHON -m pip install -r requirements.txt --quiet

# -----------------------------------------------------------
# 4. Extraer dataset de Kaggle (archive.zip → data/raw/)
# -----------------------------------------------------------
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo ""
    echo "📂  Extrayendo dataset de partidos (archive.zip)..."
    if [ ! -f "archive.zip" ]; then
        echo "❌  No se encontró archive.zip. Asegúrate de que está en la raíz del proyecto."
        read -p "Pulsa Enter para salir..."
        exit 1
    fi
    $PYTHON scripts/setup_data.py
    echo "✅  Dataset extraído."
fi

# -----------------------------------------------------------
# 5. Extraer datos xG/xGA (archive(1).zip → data/xg.csv)
# -----------------------------------------------------------
if [ ! -f "data/xg.csv" ]; then
    echo ""
    echo "📂  Extrayendo datos xG/xGA (archive(1).zip)..."
    if [ ! -f "archive(1).zip" ]; then
        echo "⚠️   No se encontró archive(1).zip. Se omiten los datos xG/xGA (opcional)."
    else
        # Descomprimir en una carpeta temporal y copiar el CSV a data/xg.csv
        mkdir -p data/tmp_xg
        unzip -o "archive(1).zip" -d data/tmp_xg > /dev/null 2>&1
        # Buscar cualquier CSV dentro y usarlo como xg.csv
        XG_FILE=$(find data/tmp_xg -name "*.csv" | head -1)
        if [ -n "$XG_FILE" ]; then
            cp "$XG_FILE" data/xg.csv
            echo "✅  xG/xGA copiado a data/xg.csv"
        else
            echo "⚠️   No se encontró un CSV dentro de archive(1).zip. Se omiten xG/xGA."
        fi
        rm -rf data/tmp_xg
    fi
fi

# -----------------------------------------------------------
# 6. Descargar ranking FIFA (si no existe snapshot previo)
# -----------------------------------------------------------
if [ ! -f "models/fifa_snapshot.csv" ]; then
    echo ""
    echo "🌐  Descargando ranking FIFA..."
    $PYTHON scripts/fetch_fifa.py
    echo "✅  Ranking FIFA descargado."
fi

# -----------------------------------------------------------
# 7. Entrenar modelo (si no existen los modelos)
# -----------------------------------------------------------
if [ ! -f "models/ml_1x2.pkl" ]; then
    echo ""
    echo "🧠  Entrenando modelo (puede tardar 1-2 minutos)..."
    $PYTHON scripts/train.py
    echo "✅  Modelo entrenado."
fi

# -----------------------------------------------------------
# 8. Lanzar el dashboard
# -----------------------------------------------------------
echo ""
echo "🚀  Iniciando el dashboard... se abrirá en tu navegador."
echo "    URL: http://localhost:8501"
echo "    Para cerrar: Ctrl+C"
echo ""
$PYTHON -m streamlit run app/dashboard.py

echo ""
echo "El servidor se ha detenido."
