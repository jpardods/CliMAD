from __future__ import annotations

from pathlib import Path
import math
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -----------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "meteo_all.csv"
OUTPUT_PATH = DATA_DIR / "meteo_all_predictions.csv"

# Fechas clave
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-10-31")
PRED_START = pd.Timestamp("2025-11-01")
PRED_END = pd.Timestamp("2026-12-31")

# Variables a predecir
TARGET_COLUMNS = [
    "precipitation",
    "pressure",
    "relative_humidity",
    "solar_radiation",
    "temperature",
    "wind_direction",
    "wind_speed",
]


# -----------------------------------------------------------
# CARGA Y FEATURES
# -----------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Carga meteo_all.csv, se asegura de que existan lat, lon, altitude e is_pred."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el fichero de entrada: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Aseguramos columnas de coordenadas (por si acaso)
    for col in ["lat", "lon", "altitude"]:
        if col not in df.columns:
            df[col] = pd.NA

    # is_pred = False para todo el histórico
    if "is_pred" not in df.columns:
        df["is_pred"] = False
    else:
        df["is_pred"] = False

    return df


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Añade dayofyear, month, sin/cos estacional."""
    df = df.copy()
    df["dayofyear"] = df[date_col].dt.dayofyear
    df["month"] = df[date_col].dt.month
    df["sin_doy"] = df["dayofyear"].apply(
        lambda x: math.sin(2 * math.pi * x / 365.25)
    )
    df["cos_doy"] = df["dayofyear"].apply(
        lambda x: math.cos(2 * math.pi * x / 365.25)
    )
    return df


def add_lag_features_per_station(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Añade lags de 1, 7 y 30 días para target_col,
    calculados por estación.
    """
    df = df.sort_values(["station_id", "date"]).copy()
    for lag in [1, 7, 30]:
        df[f"lag{lag}"] = (
            df.groupby("station_id")[target_col].shift(lag)
        )
    return df


def encode_station(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Codifica station_id como categoría entera station_id_cat."""
    df = df.copy()
    stations = sorted(df["station_id"].unique())
    station_to_cat = {sid: i for i, sid in enumerate(stations)}
    df["station_id_cat"] = df["station_id"].map(station_to_cat).astype(int)
    return df, station_to_cat


# -----------------------------------------------------------
# MODELO GLOBAL POR VARIABLE
# -----------------------------------------------------------

def train_global_model_for_variable(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[RandomForestRegressor | None, float | None]:
    """
    Entrena un modelo global RandomForest para una variable (todas las estaciones).
    Incluye:
      - lags
      - features temporales
      - station_id_cat
      - lat, lon, altitude
    """
    # Solo filas con target y dentro de train+test
    df_var = df[
        df[target_col].notna()
        & (df["date"] <= TEST_END)
    ].copy()

    if df_var.empty:
        return None, None

    # Lags por estación
    df_var = add_lag_features_per_station(df_var, target_col)
    # Features temporales
    df_var = add_time_features(df_var, "date")

    # Columnas de coordenadas deben existir
    for col in ["lat", "lon", "altitude"]:
        if col not in df_var.columns:
            df_var[col] = pd.NA

    # Eliminamos filas sin lags
    df_var = df_var.dropna(subset=["lag1", "lag7", "lag30"]).copy()
    if df_var.empty:
        return None, None

    # Split temporal
    mask_train = df_var["date"] <= TRAIN_END
    mask_test = (df_var["date"] >= TEST_START) & (df_var["date"] <= TEST_END)

    df_train = df_var[mask_train]
    df_test = df_var[mask_test]

    if df_train.empty or df_test.empty:
        return None, None

    feature_cols = [
        "lag1",
        "lag7",
        "lag30",
        "dayofyear",
        "month",
        "sin_doy",
        "cos_doy",
        "station_id_cat",
        "lat",
        "lon",
        "altitude",
    ]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    return model, mae


# -----------------------------------------------------------
# FORECAST ITERATIVO FUTURO
# -----------------------------------------------------------

def build_series_dicts(
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Dict[pd.Timestamp, float]]:
    """
    Construye:
        station_id -> { fecha -> valor_target }
    solo hasta TEST_END (train+test).
    """
    series_by_station: Dict[str, Dict[pd.Timestamp, float]] = {}
    for sid, group in df[df["date"] <= TEST_END].groupby("station_id"):
        s = (
            group[["date", target_col]]
            .dropna()
            .drop_duplicates(subset=["date"])
            .set_index("date")[target_col]
        )
        series_by_station[sid] = s.to_dict()
    return series_by_station


def make_future_features_for_date(
    date: pd.Timestamp,
    sid: str,
    series_dict: Dict[pd.Timestamp, float],
    station_to_cat: Dict,
    lat_map: Dict,
    lon_map: Dict,
    alt_map: Dict,
) -> Dict[str, float] | None:
    """
    Construye features para una fecha futura y una estación:
    lags + tiempo + station_id_cat + lat/lon/altitude.
    Si falta alguna lag, devuelve None.
    """
    d1 = date - pd.Timedelta(days=1)
    d7 = date - pd.Timedelta(days=7)
    d30 = date - pd.Timedelta(days=30)

    if d1 not in series_dict or d7 not in series_dict or d30 not in series_dict:
        return None

    dayofyear = date.timetuple().tm_yday
    month = date.month
    sin_doy = math.sin(2 * math.pi * dayofyear / 365.25)
    cos_doy = math.cos(2 * math.pi * dayofyear / 365.25)

    feats = {
        "lag1": series_dict[d1],
        "lag7": series_dict[d7],
        "lag30": series_dict[d30],
        "dayofyear": dayofyear,
        "month": month,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
        "station_id_cat": station_to_cat[sid],
        "lat": lat_map.get(sid, float("nan")),
        "lon": lon_map.get(sid, float("nan")),
        "altitude": alt_map.get(sid, float("nan")),
    }
    return feats


def forecast_global_variable(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    target_col: str,
    station_to_cat: Dict,
) -> pd.DataFrame:
    """
    Genera predicciones para target_col del 01/11/2025 al 31/12/2026
    para todas las estaciones, usando un modelo global.
    Devuelve un DataFrame con columnas: date, station_id, target_col.
    """
    future_dates = list(pd.date_range(PRED_START, PRED_END, freq="D"))
    stations = sorted(df["station_id"].unique())

    # Diccionario de series históricas por estación
    series_by_station = build_series_dicts(df, target_col)

    # Mapas de coordenadas por estación
    base_station_info = (
        df[["station_id", "lat", "lon", "altitude"]]
        .drop_duplicates(subset=["station_id"])
        .set_index("station_id")
    )
    lat_map = base_station_info["lat"].to_dict()
    lon_map = base_station_info["lon"].to_dict()
    alt_map = base_station_info["altitude"].to_dict()

    preds_rows = []

    feature_cols = [
        "lag1",
        "lag7",
        "lag30",
        "dayofyear",
        "month",
        "sin_doy",
        "cos_doy",
        "station_id_cat",
        "lat",
        "lon",
        "altitude",
    ]

    for sid in stations:
        series_dict = series_by_station.get(sid, {})
        if not series_dict:
            continue

        for date in future_dates:
            feats = make_future_features_for_date(
                date,
                sid,
                series_dict,
                station_to_cat,
                lat_map,
                lon_map,
                alt_map,
            )
            if feats is None:
                continue

            X = pd.DataFrame([feats], columns=feature_cols)
            y_hat = float(model.predict(X)[0])

            preds_rows.append(
                {
                    "date": date,
                    "station_id": sid,
                    target_col: y_hat,
                }
            )

            # Actualizamos la serie para usar esta predicción como lag
            series_dict[date] = y_hat

    if not preds_rows:
        return pd.DataFrame(columns=["date", "station_id", target_col])

    df_preds = pd.DataFrame(preds_rows)
    return df_preds


# -----------------------------------------------------------
# PIPELINE PRINCIPAL
# -----------------------------------------------------------

def main() -> None:
    df = load_data()

    # Aseguramos que las columnas target existan
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Codificamos station_id en todo el DF
    df, station_to_cat = encode_station(df)

    print("=== Entrenando modelos globales por variable (con lat/lon/altitude) ===")

    future_all_vars: List[pd.DataFrame] = []

    for target_col in TARGET_COLUMNS:
        print(f"\nVariable: {target_col}")

        if df[target_col].notna().sum() < 50:
            print("  Muy pocos datos, se omite esta variable.")
            continue

        model, mae = train_global_model_for_variable(df, target_col)

        if model is None:
            print("  No se ha podido entrenar modelo (sin datos suficientes).")
            continue

        print(f"  MAE en test (2025-01-01 a 2025-10-31): {mae:.3f}")

        df_preds = forecast_global_variable(
            model,
            df,
            target_col,
            station_to_cat,
        )

        if df_preds.empty:
            print("  No se han podido generar predicciones futuras para esta variable.")
            continue

        future_all_vars.append(df_preds)

    if not future_all_vars:
        print("No se han generado predicciones para ninguna variable.")
        return

    # Combinamos todas las variables predichas en un solo DF futuro
    df_future = future_all_vars[0]
    for df_var in future_all_vars[1:]:
        df_future = df_future.merge(
            df_var,
            on=["date", "station_id"],
            how="outer",
        )

    # Añadimos station_name, coords e is_pred
    station_info = (
        df[["station_id", "station_name", "lat", "lon", "altitude"]]
        .drop_duplicates(subset=["station_id"])
        .set_index("station_id")
    )
    df_future["station_name"] = df_future["station_id"].map(
        station_info["station_name"].to_dict()
    )
    df_future["lat"] = df_future["station_id"].map(
        station_info["lat"].to_dict()
    )
    df_future["lon"] = df_future["station_id"].map(
        station_info["lon"].to_dict()
    )
    df_future["altitude"] = df_future["station_id"].map(
        station_info["altitude"].to_dict()
    )
    df_future["is_pred"] = True

    # Orden de columnas
    id_cols = ["date", "station_id", "station_name"]
    coord_cols = ["lat", "lon", "altitude"]
    num_cols = [c for c in TARGET_COLUMNS if c in df_future.columns]
    df_future = df_future[id_cols + coord_cols + num_cols + ["is_pred"]]

    # Históricos hasta TEST_END
    df_hist = df[df["date"] <= TEST_END].copy()
    df_hist["is_pred"] = False

    # Juntamos histórico + futuro
    df_all = pd.concat([df_hist, df_future], ignore_index=True)
    df_all = df_all.sort_values(["station_id", "date"]).reset_index(drop=True)

    # Guardamos
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUTPUT_PATH, index=False)

    print("\nGuardado fichero de predicciones:")
    print(f"  {OUTPUT_PATH.resolve()}")
    print(f"Total filas: {len(df_all)}")
    print(
        f"Rango de fechas: {df_all['date'].min().date()} -> {df_all['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
