from __future__ import annotations

from pathlib import Path
import pandas as pd


# -----------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAW_FILES = [
    "meteo23.csv",
    "meteo24.csv",
    "meteo25.csv",
]

COORDS_FILE = DATA_DIR / "Estaciones_control_datos_meteorologicos.xls"

# Mapeo de código MAGNITUD → nombre de variable
MAGNITUD_TO_VARIABLE = {
    81: "wind_speed",         # Velocidad del viento (m/s)
    82: "wind_direction",     # Dirección del viento (grados)
    83: "temperature",        # Temperatura (ºC)
    86: "relative_humidity",  # Humedad relativa (%)
    87: "pressure",           # Presión barométrica (mb)
    88: "solar_radiation",    # Radiación solar (W/m2)
    89: "precipitation",      # Precipitación (l/m2)
}

# Mapa de códigos de estación → nombre (desde el anexo del PDF)
STATION_NAME_MAP = {
    "28079102": "J.M.D. Moratalaz",
    "28079103": "J.M.D. Villaverde",
    "28079104": "E.D.A.R. La China",
    "28079106": "Centro Mpal. De Acústica",
    "28079107": "J.M.D. Hortaleza",
    "28079108": "Peñagrande",
    "28079109": "J.M.D.Chamberí",
    "28079110": "J.M.D.Centro",
    "28079111": "J.M.D.Chamartin",
    "28079112": "J.M.D.Vallecas 1",
    "28079113": "J.M.D.Vallecas 2",
    "28079114": "Matadero 01",
    "28079115": "Matadero 02",
    "28079004": "Plaza España",
    "28079008": "Escuelas Aguirre",
    "28079016": "Arturo Soria",
    "28079018": "Farolillo",
    "28079024": "Casa de Campo",
    "28079035": "Plaza del Carmen",
    "28079036": "Moratalaz",
    "28079038": "Cuatro Caminos",
    "28079039": "Barrio del Pilar",
    "28079054": "Ensanche de Vallecas",
    "28079056": "Plaza Elíptica",
    "28079058": "El Pardo",
    "28079059": "Juan Carlos I",
}

OUTPUT_PATH = DATA_DIR / "meteo_all.csv"


# -----------------------------------------------------------
# CARGA Y TRANSFORMACIÓN
# -----------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Carga y concatena los CSV brutos (2023, 2024, 2025)."""
    frames = []
    for fname in RAW_FILES:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"No se encuentra el fichero: {path}")
        df = pd.read_csv(path, sep=";")
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    return raw


def expand_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pasa del formato mensual (columnas D01..D31, V01..V31) a diario.

    - Solo conserva datos con validación 'V'.
    - Crea una fila por fecha (date) y mantiene info de estación y magnitud.
    """
    day_frames = []

    for day in range(1, 32):
        dcol = f"D{day:02d}"
        vcol = f"V{day:02d}"
        if dcol not in df.columns or vcol not in df.columns:
            continue

        base_cols = [
            "PROVINCIA",
            "MUNICIPIO",
            "ESTACION",
            "MAGNITUD",
            "PUNTO_MUESTREO",
            "ANO",
            "MES",
        ]
        cols = [c for c in base_cols if c in df.columns] + [dcol, vcol]

        sub = df[cols].copy()

        # Solo datos válidos
        sub = sub[sub[vcol] == "V"].copy()

        # Renombramos valor
        sub = sub.rename(columns={dcol: "value"})

        # Construimos la fecha (los días imposibles tipo 31/02 se ponen NaT)
        sub["date"] = pd.to_datetime(
            dict(year=sub["ANO"], month=sub["MES"], day=day),
            errors="coerce",
        )
        sub = sub[~sub["date"].isna()].copy()

        sub["day"] = day
        sub = sub.drop(columns=[vcol])

        day_frames.append(sub)

    long_df = pd.concat(day_frames, ignore_index=True)
    return long_df


def build_station_id_and_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea:
      - station_id tipo 28079004
      - station_name usando el mapa del anexo (si no existe, usa el id)
    """
    # Código fallback: 2 dígitos provincia + 3 municipio + 3 estación
    fallback_code = (
        df["PROVINCIA"].astype(int).astype(str).str.zfill(2)
        + df["MUNICIPIO"].astype(int).astype(str).str.zfill(3)
        + df["ESTACION"].astype(int).astype(str).str.zfill(3)
    )

    # Si PUNTO_MUESTREO existe, lo usamos; si no, tiramos del fallback
    station_raw = df["PUNTO_MUESTREO"].where(
        df["PUNTO_MUESTREO"].notna(),
        fallback_code,
    )

    # PUNTO_MUESTREO viene como "28079102_81_98" → nos quedamos con "28079102"
    df["station_id"] = station_raw.astype(str).str.split("_").str[0]

    # Nombre de estación: del mapa, o el propio id si no lo tenemos
    df["station_name"] = df["station_id"].map(STATION_NAME_MAP).fillna(
        df["station_id"]
    )

    return df


def map_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduce MAGNITUD a nombres de variable (temperature, humidity, etc.)
    y elimina magnitudes que no estén en MAGNITUD_TO_VARIABLE.
    """
    df["variable"] = df["MAGNITUD"].map(MAGNITUD_TO_VARIABLE)
    df = df[~df["variable"].isna()].copy()
    return df


def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivota de formato largo:

        date, station_id, station_name, variable, value

    a formato ancho:

        date, station_id, station_name, temperature, wind_speed, ...

    """
    pivot = df.pivot_table(
        index=["date", "station_id", "station_name"],
        columns="variable",
        values="value",
        aggfunc="mean",
    ).reset_index()

    # Ordenamos columnas: primero identificadores, luego variables
    id_cols = ["date", "station_id", "station_name"]
    variable_cols = [c for c in pivot.columns if c not in id_cols]
    pivot = pivot[id_cols + variable_cols]

    # Ordenamos por estación y fecha
    pivot = pivot.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Marcamos todo como dato observado (no predicción)
    pivot["is_pred"] = False

    return pivot


# -----------------------------------------------------------
# COORDENADAS DE ESTACIONES
# -----------------------------------------------------------

def load_station_coords() -> pd.DataFrame | None:
    """
    Lee el Excel de estaciones y extrae:
        station_id, lat, lon, altitude
    Ajusta los nombres de columna a los reales de tu Excel.
    """
    if not COORDS_FILE.exists():
        print(f"[AVISO] No se ha encontrado el fichero de coordenadas: {COORDS_FILE}")
        return None

    coords = pd.read_excel(COORDS_FILE)

    # 👇 AJUSTA ESTOS NOMBRES A LOS DE TU EXCEL 👇
    STATION_ID_COL = "CÓDIGO"   # ej: "CODIGO_ESTACION", "CODIGO", etc.
    LAT_COL = "LATITUD"                  # ej: "LATITUD", "Latitud", ...
    LON_COL = "LONGITUD"                 # ej: "LONGITUD", "Longitud", ...
    ALT_COL = "ALTITUD"                   # ej: "ALTURA", "Altura", "ELEVACION", etc.

    # Creamos station_id en formato uniforme
    coords["station_id"] = coords[STATION_ID_COL].astype(str).str.zfill(8)

    # Seleccionamos columnas
    coords = coords[["station_id", LAT_COL, LON_COL, ALT_COL]].rename(
        columns={
            LAT_COL: "lat",
            LON_COL: "lon",
            ALT_COL: "altitude"
        }
    )

    coords = coords.drop_duplicates(subset=["station_id"]).reset_index(drop=True)

    return coords



# -----------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------

def build_meteo_all() -> pd.DataFrame:
    """Pipeline completo para construir el dataframe meteo_all con coordenadas."""
    raw = load_raw_data()
    daily = expand_to_daily(raw)
    daily = build_station_id_and_name(daily)
    daily = map_variables(daily)

    daily_short = daily[["date", "station_id", "station_name", "variable", "value"]]

    meteo_all = pivot_to_wide(daily_short)

    # --- Añadimos coordenadas de estaciones ---
    coords = load_station_coords()
    if coords is not None:
        meteo_all = meteo_all.merge(coords, on="station_id", how="left")

        # Reordenamos columnas: ids + coords + variables + is_pred
        id_cols = ["date", "station_id", "station_name"]
        extra_cols = ["lat", "lon", "altitude"]
        other_cols = [c for c in meteo_all.columns
                      if c not in id_cols + extra_cols]
        meteo_all = meteo_all[id_cols + extra_cols + other_cols]

    return meteo_all


def save_meteo_all(path: Path | str = OUTPUT_PATH) -> None:
    """Genera y guarda meteo_all.csv."""
    df = build_meteo_all()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Guardado fichero combinado: {path} ({len(df)} filas)")
    print(f"Columnas: {list(df.columns)}")


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    save_meteo_all()
