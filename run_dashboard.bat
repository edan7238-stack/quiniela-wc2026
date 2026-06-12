@echo off
REM ============================================================
REM  Lanzador del dashboard del Modelo Probabilistico WC 2026.
REM  Doble clic para abrir el programa en tu navegador.
REM  Para cerrarlo: pulsa Ctrl+C aqui o cierra esta ventana.
REM ============================================================
cd /d "%~dp0"

REM Evita el aviso de correo de Streamlit la primera vez.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
    >"%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >>"%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo  Iniciando el dashboard... se abrira solo en tu navegador.
echo  URL: http://localhost:8501
echo.

python -m streamlit run "app\dashboard.py"

echo.
echo  El servidor se ha detenido.
pause
