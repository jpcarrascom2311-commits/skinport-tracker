import csv
import subprocess
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import numpy as np

BEST_FIELD = "median_7d"  # si quiere más estabilidad, use "median_30d"
API = "https://api.skinport.com/v1/sales/history"
HERE = Path(__file__).resolve().parent
HISTORY_CSV = HERE / "skinport_history.csv"
LATEST_CSV  = HERE / "skinport_latest.csv"
BEST_CSV    = HERE / "skinport_best.csv"
COST_BASIS = {
    "AUG | Death by Puppy (Minimal Wear)": 5.74,
    "★ Broken Fang Gloves | Needle Point (Well-Worn)": 64.05,
    "Desert Eagle | Kumicho Dragon (Minimal Wear)": 31.75,
    "Galil AR | Chromatic Aberration (Minimal Wear)": 8.08,
    "M4A4 | Temukau (Field-Tested)": 34.84,
    "MP7 | Smoking Kills (Minimal Wear)": 15.34,
    "MP9 | Latte Rush (Field-Tested)": 33.69,
    "★ Paracord Knife | Slaughter (Factory New)": 159.64,
    "Special Agent Ava | FBI": 22.17,
    "StatTrak™ AK-47 | Nightwish (Minimal Wear)": 55.40,
    "StatTrak™ AWP | Fever Dream (Minimal Wear)": 25.70,
    "StatTrak™ SG 553 | Darkwing (Minimal Wear)": 1.92,
    "Tec-9 | Jambiya (Minimal Wear)": 0.86,
    "The Elite Mr. Muhlik | Elite Crew": 12.86,
    "XM1014 | Run Run Run (Factory New)": 3.85,
}

ITEMS = [
    "M4A4 | Temukau (Field-Tested)",
    "Special Agent Ava | FBI",
    "The Elite Mr. Muhlik | Elite Crew",
    "Desert Eagle | Kumicho Dragon (Minimal Wear)",
    "★ Paracord Knife | Slaughter (Factory New)",
    "MP7 | Smoking Kills (Minimal Wear)",
    "StatTrak™ AWP | Fever Dream (Minimal Wear)",
    "MP9 | Latte Rush (Field-Tested)",
    "★ Broken Fang Gloves | Needle Point (Well-Worn)",
    "StatTrak™ SG 553 | Darkwing (Minimal Wear)",
    "StatTrak™ AK-47 | Nightwish (Minimal Wear)",
    "Tec-9 | Jambiya (Minimal Wear)",
    "Galil AR | Chromatic Aberration (Minimal Wear)",
    "AUG | Death by Puppy (Minimal Wear)",
    "XM1014 | Run Run Run (Factory New)",
]

PARAMS_BASE = {
    "app_id": 730,
    "currency": "USD",
}

HEADERS = {
    "Accept-Encoding": "br",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}

# Archivos en la misma carpeta del script
HERE = Path(__file__).resolve().parent
HISTORY_CSV = HERE / "skinport_history.csv"
BEST_CSV = HERE / "skinport_best.csv"

# Columna a usar para "mejor precio" (robusto ante manipulación)
# Recomendación: median_7d o median_30d (más estables que 24h).
BEST_FIELD = "median_7d"


def get_json(url: str, params: dict) -> list[dict]:
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def flat_period(obj: dict, key: str):
    p = obj.get(key) or {}
    return (p.get("min"), p.get("max"), p.get("avg"), p.get("median"), p.get("volume"))


def write_latest_and_append_history(rows: list[dict]) -> None:
    ts = datetime.now().isoformat(timespec="seconds")

    header = [
        "timestamp", "market_hash_name", "currency",
        "min_24h","max_24h","avg_24h","median_24h","vol_24h",
        "min_7d","max_7d","avg_7d","median_7d","vol_7d",
        "min_30d","max_30d","avg_30d","median_30d","vol_30d",
        "min_90d","max_90d","avg_90d","median_90d","vol_90d",
        "item_page","market_page"
    ]

    latest_rows = []
    for x in rows:
        latest_rows.append([
            ts,
            x.get("market_hash_name"),
            x.get("currency"),
            *flat_period(x, "last_24_hours"),
            *flat_period(x, "last_7_days"),
            *flat_period(x, "last_30_days"),
            *flat_period(x, "last_90_days"),
            x.get("item_page"),
            x.get("market_page"),
        ])

    # 1) SNAPSHOT (se pisa cada vez)
    with open(LATEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(latest_rows)

    # 2) HISTÓRICO (append), pero creando header si falta o si está malo
    need_header = True
    if HISTORY_CSV.exists():
        with open(HISTORY_CSV, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            need_header = ("timestamp" not in first)  # header inválido

    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerows(latest_rows)



def robust_best_from_series(values: np.ndarray) -> float | None:
    """Devuelve un valor representativo robusto (anti-outliers).
    - elimina NaN
    - filtro IQR
    - retorna el valor más cercano a la mediana dentro del rango filtrado
    """
    v = values.astype(float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    if v.size < 5:
        return float(np.median(v))

    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    vf = v[(v >= low) & (v <= high)]
    if vf.size == 0:
        return float(np.median(v))

    med = float(np.median(vf))
    best = float(vf[np.argmin(np.abs(vf - med))])
    return best


def robust_pick_closest(values: np.ndarray, target: float) -> float | None:
    v = values.astype(float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None

    # Ventana chica: elegimos el más cercano, sin estadística compleja
    if v.size < 5:
        return float(v[np.argmin(np.abs(v - target))])

    # Filtro IQR anti-outliers (transferencias/manipulación)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    vf = v[(v >= low) & (v <= high)]

    # Si el filtro dejó vacío, usar todo (fallback)
    if vf.size == 0:
        vf = v

    return float(vf[np.argmin(np.abs(vf - target))])


def build_best_csv(HISTORY_CSV, BEST_CSV, max_rows_per_item: int = 60):
    # 1) Leer CSV con tolerancia a archivos "sucios"
    df = pd.read_csv(HISTORY_CSV)

    # 2) Validaciones mínimas de columnas
    required_cols = {"timestamp", "market_hash_name", BEST_FIELD}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Faltan columnas en {HISTORY_CSV}: {sorted(missing)}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # 3) Normalizar strings para mejorar el match con COST_BASIS
    df["market_hash_name"] = (
        df["market_hash_name"]
        .astype(str)
        .str.strip()
    )

    # 4) Parsear timestamp de forma segura
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        # No fallamos: solo advertimos y eliminamos filas rotas
        print(f"WARN: {bad_ts} filas con timestamp inválido; se omiten.")
        df = df.dropna(subset=["timestamp"])

    # 5) Convertir campo objetivo a numérico
    df[BEST_FIELD] = pd.to_numeric(df[BEST_FIELD], errors="coerce")

    # Si el campo objetivo es todo NaN, es mejor fallar temprano
    if df[BEST_FIELD].notna().sum() == 0:
        raise RuntimeError(
            f"El campo {BEST_FIELD} quedó vacío (NaN) tras parseo. "
            f"Revise que exista y que tenga números en {HISTORY_CSV}."
        )

    # 6) Debug de match COST_BASIS vs datos
    data_items = set(df["market_hash_name"].unique())
    cost_items = set(COST_BASIS.keys())

    missing_cost = sorted(data_items - cost_items)
    extra_cost = sorted(cost_items - data_items)

    if missing_cost:
        print("WARN: hay items en el CSV sin costo en COST_BASIS (best será None):")
        for x in missing_cost[:20]:
            print("  -", x)
        if len(missing_cost) > 20:
            print(f"  ... +{len(missing_cost)-20} más")

    if extra_cost:
        print("WARN: hay items en COST_BASIS que no aparecen en el CSV (revisar nombres exactos):")
        for x in extra_cost[:20]:
            print("  -", x)
        if len(extra_cost) > 20:
            print(f"  ... +{len(extra_cost)-20} más")

    # 7) Calcular best por item
    out_rows = []

    for item, g in df.groupby("market_hash_name", sort=False):
        g = g.sort_values("timestamp").tail(max_rows_per_item)

        cost = COST_BASIS.get(item)
        series = g[BEST_FIELD].to_numpy()

        last_row = g.iloc[-1]
        last_val = last_row[BEST_FIELD]
        last_ts = last_row["timestamp"]

        # isoformat seguro
        last_ts_str = (
            last_ts.isoformat(timespec="seconds")
            if hasattr(last_ts, "isoformat") and pd.notna(last_ts)
            else str(last_ts)
        )

        if cost is None or pd.isna(cost):
            best = None
            dist = None
            method = "no_cost_basis"
        else:
            best = robust_pick_closest(series, float(cost))
            dist = None if best is None else abs(best - float(cost))
            method = f"closest_to_cost_{BEST_FIELD}_iqr"

        out_rows.append({
            "market_hash_name": item,
            "cost_basis": None if cost is None else float(cost),
            "best_field": BEST_FIELD,
            "best_value": None if best is None else round(best, 2),
            "distance_to_cost": None if dist is None else round(dist, 2),
            "last_value": None if pd.isna(last_val) else round(float(last_val), 2),
            "last_timestamp": last_ts_str,
            "method": method,
            "n_samples": int(pd.Series(series).notna().sum()),
        })

    out = pd.DataFrame(out_rows)

    # 8) Orden útil: primero los que sí tienen best
    out["has_best"] = out["best_value"].notna()
    out = out.sort_values(["has_best", "distance_to_cost", "market_hash_name"], ascending=[False, True, True])
    out = out.drop(columns=["has_best"])

    out.to_csv(BEST_CSV, index=False, encoding="utf-8")
    print(f"OK: best -> {BEST_CSV} (field={BEST_FIELD}) filas={len(out)}")




import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIT_EXE = r"C:\Program Files\Git\cmd\git.exe"  # ajuste si Git está en otra ruta

def run_git(args):
    return subprocess.run([GIT_EXE, *args], cwd=HERE, capture_output=True, text=True)

def git_sync_or_exit():
    fetch = run_git(["fetch", "origin"])
    if fetch.returncode != 0:
        print("WARN: git fetch falló.")
        print(fetch.stderr[:400])
        return False

    pull = run_git(["pull", "--rebase", "origin", "main"])
    if pull.returncode != 0:
        print("WARN: git pull --rebase falló. No se hará commit/push.")
        print(pull.stderr[:400])
        return False

    return True

import subprocess

GIT_EXE = r"C:\Program Files\Git\cmd\git.exe"  # ajuste si su Git está en otra ruta

def run_git(args):
    p = subprocess.run([GIT_EXE, *args], cwd=str(HERE), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} falló:\n{p.stderr}")
    return p

def git_commit_push():
    # Agregar archivos que quiere versionar
    run_git(["add", "skinport_latest.csv", "skinport_best.csv", "skinport_track.py"])

    staged = run_git(["diff", "--cached", "--name-only"]).stdout.strip()
    if not staged:
        print("INFO: no hay cambios staged para commit.")
        return

    print("DEBUG: staged:\n" + staged)

    run_git(["commit", "-m", "update prices"])
    run_git(["push", "origin", "main"])
    print("OK: git push exitoso")


def git_sync_or_exit():
    # si hay cambios, los guardamos temporalmente
    dirty = run_git(["status", "--porcelain"])
    had_changes = bool(dirty.stdout.strip())

    if had_changes:
        st = run_git(["stash", "push", "-u", "-m", "auto-stash before rebase"])
        if st.returncode != 0:
            print("WARN: git stash falló.")
            print(st.stderr[:400])
            return False

    pull = run_git(["pull", "--rebase", "origin", "main"])
    if pull.returncode != 0:
        print("WARN: git pull --rebase falló. No se hará commit/push.")
        print(pull.stderr[:400])
        return False

    if had_changes:
        pop = run_git(["stash", "pop"])
        if pop.returncode != 0:
            print("WARN: stash pop falló (puede haber conflictos).")
            print(pop.stderr[:400])
            return False

    return True


def main():
    if not ITEMS:
        raise RuntimeError("ITEMS está vacío.")

    params = dict(PARAMS_BASE)
    params["market_hash_name"] = ",".join(ITEMS)

    data = get_json(API, params)

    write_latest_and_append_history(data)
    print(f"OK: {len(data)} items -> {LATEST_CSV.name} (snapshot) + {HISTORY_CSV.name} (append)")
    print(f"OK: {len(data)} items -> {HISTORY_CSV.name} (append)")

    build_best_csv(HISTORY_CSV, BEST_CSV,max_rows_per_item=60)
    print(f"OK: best -> {BEST_CSV.name} (field={BEST_FIELD})")


if __name__ == "__main__":
    if not git_sync_or_exit():
        raise SystemExit(1)

    main()  # aquí recién se crean/modifican CSV

    build_best_csv(HISTORY_CSV, BEST_CSV, max_rows_per_item=60)

    git_commit_push()


