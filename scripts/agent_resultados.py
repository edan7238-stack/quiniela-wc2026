"""Agente de ingreso de resultados — lee marcadores de CAPTURAS y/o ENLACES y los registra.

Usa la API de Anthropic (Tool Use + visión, modelo `claude-sonnet-4-6`) para extraer los
resultados FINALIZADOS de partidos del Mundial y escribirlos en `data/wc2026_results.csv`
(lo mismo que la página "Ingreso de datos" del dashboard), vía `src/results_io.add_wc_result`.

Requisitos:
    pip install -r requirements.txt
    setx ANTHROPIC_API_KEY "sk-ant-..."   (Windows; reabre la terminal)

Uso:
    python scripts/agent_resultados.py --image captura1.png captura2.png
    python scripts/agent_resultados.py --url https://sitio/resultados --dry-run
    python scripts/agent_resultados.py --image c.png --retrain
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import results_io

MODEL = "claude-sonnet-4-6"
MAX_STEPS = 24

SYSTEM = (
    "Eres un asistente que registra resultados FINALIZADOS (FT) de partidos del Mundial 2026 "
    "en el programa del usuario. Reglas:\n"
    "- Registra SOLO partidos terminados (marcador final). Ignora los programados, en juego, "
    "aplazados o CANCELADOS.\n"
    "- El equipo listado primero / arriba es el LOCAL.\n"
    "- Usa nombres canónicos; ante la duda llama a `list_valid_teams` (p. ej. 'USA' = 'United "
    "States', 'Korea Republic' = 'South Korea').\n"
    "- Fechas en YYYY-MM-DD. Las capturas suelen mostrar DD/MM/YY (p. ej. '09/06/26' = 2026-06-09).\n"
    "- Registra cada partido con `add_match_result`. Si una herramienta devuelve error con "
    "sugerencias, corrige el nombre y reintenta. No reintentes los duplicados.\n"
    "- Al terminar, resume en una frase cuántos resultados registraste."
)

TOOLS = [
    {
        "name": "list_valid_teams",
        "description": "Lista los nombres canónicos válidos de las 48 selecciones del Mundial 2026. "
                       "Úsala para mapear variantes ('USA'->'United States') antes de registrar.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_match_result",
        "description": "Registra el resultado FINALIZADO de un partido del Mundial. El equipo "
                       "listado primero es el LOCAL. Devuelve error con sugerencias si un nombre "
                       "no se reconoce.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Fecha del partido en formato YYYY-MM-DD."},
                "home_team": {"type": "string", "description": "Equipo local (el listado primero)."},
                "away_team": {"type": "string", "description": "Equipo visitante."},
                "home_score": {"type": "integer", "description": "Goles del local."},
                "away_score": {"type": "integer", "description": "Goles del visitante."},
            },
            "required": ["date", "home_team", "away_team", "home_score", "away_score"],
        },
    },
]

WEB_FETCH = {"type": "web_fetch_20260209", "name": "web_fetch"}

_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".gif": "image/gif", ".webp": "image/webp"}


def _media_type(path: str) -> str:
    return _MEDIA.get(Path(path).suffix.lower(), "image/png")


def _execute(name: str, tool_input: dict, dry_run: bool) -> dict:
    if name == "list_valid_teams":
        return {"ok": True, "teams": results_io.valid_teams()}
    if name == "add_match_result":
        return results_io.add_wc_result(
            tool_input.get("date"), tool_input.get("home_team"), tool_input.get("away_team"),
            tool_input.get("home_score"), tool_input.get("away_score"), dry_run=dry_run)
    return {"ok": False, "message": f"Herramienta desconocida: {name}"}


def _initial_content(args) -> list[dict]:
    content: list[dict] = []
    for img in args.image or []:
        data = base64.standard_b64encode(Path(img).read_bytes()).decode("utf-8")
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": _media_type(img), "data": data}})
    parts = ["Extrae los resultados FINALIZADOS de los partidos del Mundial y regístralos uno a uno."]
    if args.image:
        parts.append("Las imágenes adjuntas son capturas de marcadores; el equipo de arriba es el local.")
    if args.url:
        parts.append("Abre estos enlaces con web_fetch y extrae los marcadores finales: "
                     + " ".join(args.url))
    if args.text:
        parts.append("Contexto/resultados en texto: " + args.text)
    content.append({"type": "text", "text": "\n".join(parts)})
    return content


def main() -> None:
    ap = argparse.ArgumentParser(description="Agente que ingresa resultados del Mundial.")
    ap.add_argument("--image", nargs="+", help="Capturas de pantalla con marcadores.")
    ap.add_argument("--url", nargs="+", help="Enlaces a páginas de resultados (web_fetch).")
    ap.add_argument("--text", help="Resultados/contexto en texto plano.")
    ap.add_argument("--dry-run", action="store_true", help="No escribe; solo muestra lo que haría.")
    ap.add_argument("--retrain", action="store_true",
                    help="Tras registrar, ejecuta scripts/train.py para refrescar el modelo.")
    ap.add_argument("--model", default=MODEL, help=f"Modelo de Claude (def. {MODEL}).")
    args = ap.parse_args()

    if not (args.image or args.url or args.text):
        ap.error("Da al menos una entrada: --image, --url o --text.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print('Falta ANTHROPIC_API_KEY. Configúrala con:  setx ANTHROPIC_API_KEY "sk-ant-..."  '
              "y reabre la terminal.", file=sys.stderr)
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()
    tools = TOOLS + ([WEB_FETCH] if args.url else [])
    messages = [{"role": "user", "content": _initial_content(args)}]
    written: list[dict] = []

    for _ in range(MAX_STEPS):
        resp = client.messages.create(
            model=args.model, max_tokens=8192, system=SYSTEM, tools=tools,
            messages=messages, thinking={"type": "adaptive"})
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                print("🤖", b.text.strip())

        if resp.stop_reason == "end_turn":
            break
        if resp.stop_reason == "pause_turn":      # web_fetch en curso: reenviar para continuar
            continue
        if resp.stop_reason == "tool_use":
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = _execute(b.name, dict(b.input), args.dry_run)
                    if b.name == "add_match_result" and out.get("ok"):
                        written.append(out["row"])
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out, ensure_ascii=False),
                                    "is_error": not out.get("ok", False)})
            messages.append({"role": "user", "content": results})
            continue
        print(f"(fin: stop_reason={resp.stop_reason})")
        break

    label = "que se escribirían" if args.dry_run else "registrados"
    print(f"\n{'[dry-run] ' if args.dry_run else ''}Resultados {label}: {len(written)}")
    for r in written:
        print(f"  {r['date']}  {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}")

    if written and not args.dry_run:
        if args.retrain:
            print("\nRefrescando el modelo (scripts/train.py)...")
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "train.py")],
                           check=False)
        else:
            print("\nPulsa 'Recalcular' en el dashboard (o corre `python scripts/train.py`) "
                  "para que el modelo use los nuevos resultados.")


if __name__ == "__main__":
    main()
