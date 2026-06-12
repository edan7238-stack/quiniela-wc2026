# Desplegar la Quiniela WC 2026 como app web (para iPhone/iOS y cualquier dispositivo)

> Objetivo: publicar el dashboard en **Streamlit Community Cloud** (gratis) y obtener una
> **URL pública** que cualquier usuario de iOS pueda abrir en Safari (y "Añadir a pantalla de
> inicio" para tener un icono tipo app). No requiere App Store ni instalar nada.

El proyecto ya quedó **preparado y con un commit de Git** (`main`). Solo faltan 3 cosas que
debes hacer tú porque requieren tu cuenta: (1) crear cuenta de GitHub, (2) subir el repo,
(3) conectar Streamlit Cloud. Aquí van los pasos exactos.

---

## Paso 1 — Crear cuenta de GitHub (gratis, ~3 min)

1. Entra en https://github.com/signup
2. Pon tu correo, una contraseña y un nombre de usuario (p. ej. `eduardo-quiniela`). Apúntalo.
3. Verifica tu correo cuando te llegue el código.

## Paso 2 — Crear el repositorio (vacío)

1. Entra en https://github.com/new
2. **Repository name**: `quiniela-wc2026` (o el que quieras).
3. **Visibilidad**: elige **Private** (recomendado). El repo será privado, pero la app web
   igual será pública por su URL. Así no publicas el dataset de Kaggle.
4. **NO** marques "Add a README", "Add .gitignore" ni "license" (debe quedar vacío).
5. Botón **Create repository**. Te quedará una URL tipo:
   `https://github.com/TU_USUARIO/quiniela-wc2026.git`

## Paso 3 — Subir el proyecto a GitHub

Abre **PowerShell** en la carpeta del proyecto y ejecuta (cambia `TU_USUARIO` y el nombre del
repo si lo llamaste distinto):

```powershell
cd "C:\Modelo probabilístico WC 2026"
git remote add origin https://github.com/TU_USUARIO/quiniela-wc2026.git
git push -u origin main
```

- La **primera vez**, Windows abrirá una ventana del navegador para **iniciar sesión en GitHub**
  (Git Credential Manager). Inicia sesión y autoriza. Luego el `push` continúa solo.
- Si ya estabas logueado, simplemente subirá los 56 archivos.

> (Opcional) Antes del push puedes poner tu identidad real en los commits:
> ```powershell
> git config user.name "Tu Nombre"
> git config user.email "tu-correo-de-github@ejemplo.com"
> ```

## Paso 4 — Desplegar en Streamlit Community Cloud

1. Entra en https://share.streamlit.io y pulsa **Sign in** → **Continue with GitHub**
   (usa la misma cuenta del Paso 1) y autoriza el acceso.
2. Pulsa **Create app** → **Deploy a public app from GitHub**.
3. Rellena:
   - **Repository**: `TU_USUARIO/quiniela-wc2026`
   - **Branch**: `main`
   - **Main file path**: `app/dashboard.py`
4. (Opcional) **Advanced settings** → Python version: deja la que viene, o **3.12**.
5. Pulsa **Deploy**. La primera vez tarda **2-5 minutos** (instala las librerías).
6. Al terminar tendrás una URL pública, p. ej.:
   `https://quiniela-wc2026-xxxx.streamlit.app`

## Paso 5 — Compartir y usar en iPhone (iOS)

1. Comparte esa URL por WhatsApp/donde quieras.
2. En el iPhone: abrir el enlace en **Safari** → botón **Compartir** (cuadro con flecha) →
   **Añadir a pantalla de inicio**. Queda un **icono** que abre la quiniela como si fuera una app.

---

## Actualizar la app más adelante (durante el Mundial)

Cada vez que cambies algo (p. ej. cargas resultados y reentrenas con `python scripts/train.py`),
sube los cambios y Streamlit Cloud **redesplega solo**:

```powershell
cd "C:\Modelo probabilístico WC 2026"
git add -A
git commit -m "Actualizo resultados/modelos"
git push
```

---

## Cosas a tener en cuenta (importante y honesto)

- **La app es pública**: cualquiera con el enlace puede verla y usar todas las páginas,
  incluida "Ingreso de datos". Si te preocupa que alguien toque los datos, dímelo y le añado
  una **contraseña** (pantalla de login simple).
- **Ingreso de resultados en la nube**: en Streamlit Cloud el almacenamiento es **compartido y
  temporal**: si un visitante ingresa un resultado, lo ven todos, y se **reinicia** cuando la app
  se duerme por inactividad. Para datos persistentes de verdad haría falta una base de datos
  (lo podemos añadir aparte). La forma fiable de actualizar datos es: editarlos en tu PC y hacer
  `git push` (se redesplega con los datos nuevos).
- **Arranque "frío"**: si nadie la usa un rato, la app se duerme; el primer acceso tarda unos
  segundos en despertar. Es normal en el plan gratuito.
- **El agente de IA** (resultados desde capturas) **no** corre en la web (necesita tu API key);
  la ingesta manual sí.
- **Privacidad del dataset**: por eso recomendé el repo **privado**; el código y los datos de
  Kaggle no quedan públicos, solo la app.

¿Listo? Cuando tengas la cuenta de GitHub creada, dime tu usuario y te ayudo a verificar el push
y el deploy.
