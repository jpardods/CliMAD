import pandas as pd
import datetime
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------

DATA_PATH = "src/data/meteo_all_predictions.csv"
TEMPERATURE_COL = "temperature"
# Carga de polideportivos / campos de fútbol municipales
FIELDS_PATH = "src/data/200186-0-polideportivos.csv"


# -----------------------------------------------------------
# CARGA DE DATOS
# -----------------------------------------------------------

df = pd.read_csv(DATA_PATH)
# Aseguramos tipos
df["date"] = pd.to_datetime(df["date"])
df["station_id"] = df["station_id"].astype(str)

# Rango de fechas
min_date = df["date"].min()
max_date = df["date"].max()

# Info de estaciones
station_info = (
    df[["station_id", "station_name"]]
    .drop_duplicates()
    .sort_values("station_name")
)

today = datetime.date.today()

station_options = [
    {"label": row["station_name"], "value": row["station_id"]}
    for _, row in station_info.iterrows()
]

default_station = station_options[0]["value"]
default_station2 = station_options[1]["value"] if len(station_options) > 1 else default_station

# Variables numéricas para el gráfico de comparación
exclude_cols = {
    "date",
    "station_id",
    "station_name",
    "lat",
    "lon",
    "altitude",
    "is_pred",
    "station_id_cat",
}
variable_options = [
    c for c in df.columns
    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
]

VARIABLE_LABELS = {
    "temperature": "Temperatura (°C)",
    "precipitation": "Precipitación (mm)",
    "pressure": "Presión (hPa)",
    "relative_humidity": "Humedad relativa (%)",
    "solar_radiation": "Radiación solar (W/m²)",
    "wind_speed": "Velocidad del viento (m/s)",
    "wind_direction": "Dirección del viento (°)",
}

default_variable = "temperature" if "temperature" in variable_options else variable_options[0]

# Rango de temperatura para el mapa
temp_min = df[TEMPERATURE_COL].min()
temp_max = df[TEMPERATURE_COL].max()

# Centro aproximado del mapa (Madrid)
center_lat = df["lat"].mean()
center_lon = df["lon"].mean()

polidep_df = pd.read_csv(
    FIELDS_PATH,
    encoding="latin1",
    sep=";",
)
polidep_df = polidep_df.rename(
    columns={
        "NOMBRE": "field_name",
        "LATITUD": "lat",
        "LONGITUD": "lon",
    }
)
# Limpiamos columnas que nos interesan
fields_df = polidep_df[["field_name", "lat", "lon", "NOMBRE-VIA", "CLASE-VIAL", "NUM", "EMAIL", "TELEFONO"]]
fields_df = fields_df.dropna(subset=["lat", "lon"])
fields_df = fields_df.rename(columns={
    "NOMBRE-VIA": "street",
    "NUM": "number",
    "TELEFONO": "phone",
    "CLASE-VIAL": "street_type",
    "EMAIL": "email",
})

# Garantizamos columnas aunque no existan
for col in ["street", "number", "phone", "email", "street_type"]:
    if col not in fields_df.columns:
        fields_df[col] = ""

fields_df = fields_df[["field_name", "lat", "lon", "street", "number", "phone", "email", "street_type"]]
fields_df = fields_df.dropna(subset=["lat", "lon"])

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Distancia entre dos puntos (lat, lon) en metros.
    """
    R = 6371000  # radio de la Tierra en metros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# -----------------------------------------------------------
# INICIALIZACIÓN DE LA APP
# -----------------------------------------------------------

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # para despliegue en Render


# -----------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------

app.layout = html.Div(
    className="app-container",
    children=[
        # CABECERA
        html.Div(
            className="app-header",
            children=[
                html.H1("CliMAD – Meteorología y predicciones en Madrid",
                        className="app-title"),
                html.P("Explora los datos históricos y las predicciones diarias por estación en la ciudad de Madrid.",
                       className="app-subtitle"),
            ],
        ),

        # CONTENEDOR SUPERIOR: controles + mapa + panel de datos
        html.Div(
            className="mt-3",
            children=[
                html.Div(
                    className="row g-3",
                    children=[
                        # Columna izquierda: calendario / controles
                        html.Div(
                            className="col-12 col-lg-3",
                            children=[
                                html.Div(
                                    className="card-panel",
                                    children=[
                                        html.Div("Controles", className="section-title"),
                                        html.Div("Fecha", className="control-label"),
                                        dcc.DatePickerSingle(
                                            id="date-picker",
                                            min_date_allowed=min_date,
                                            max_date_allowed=max_date,
                                            date=today if min_date <= pd.to_datetime(today) <= max_date else min_date,
                                            display_format="YYYY-MM-DD",
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Div(
                                            id="selected-station-label",
                                            style={"marginTop": "6px", "fontWeight": "bold"},
                                        ),
                                        html.P(
                                            "Haz clic en una estación del mapa para ver sus datos "
                                            "en la fecha seleccionada.",
                                            style={"fontSize": "12px", "color": "#9ca3af"},
                                        ),
                                    ],
                                )
                            ],
                        ),

                        # Columna central: mapa
                        html.Div(
                            className="col-12 col-lg-6",
                            children=[
                                html.Div(
                                    className="card-panel",
                                    children=[
                                        html.Div("Mapa de estaciones", className="section-title"),
                                        dcc.Graph(id="map-graph", config={"displayModeBar": False}),
                                    ],
                                )
                            ],
                        ),

                        # Columna derecha: panel de datos del día
                        html.Div(
                            className="col-12 col-lg-3",
                            children=[
                                html.Div(
                                    className="card-panel mb-3",
                                    children=[
                                        html.Div("Datos del día", className="section-title"),
                                        html.Div(
                                            id="station-data-panel",
                                            style={"maxHeight": "260px", "overflowY": "auto"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="card-panel",
                                    children=[
                                        html.Div("Campos de fútbol cercanos (≤ 1km)", className="section-title"),
                                        html.Div(
                                            id="nearby-fields-panel",
                                            style={"maxHeight": "220px", "overflowY": "auto", "fontSize": "12px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),

        html.Hr(style={"borderColor": "#1f2937", "marginTop": "24px", "marginBottom": "16px"}),

        # CONTENEDOR INFERIOR: comparación de estaciones
        html.Div(
            className="mt-2",
            children=[
                html.Div(
                    className="card-panel",
                    children=[
                        html.Div("Comparación de estaciones en una variable", className="section-title"),

                        html.Div(
                            className="row g-3 mb-2",
                            children=[
                                html.Div(
                                    className="col-12 col-md-4",
                                    children=[
                                        html.Div("Estación 1", className="control-label"),
                                        dcc.Dropdown(
                                            id="compare-station-1",
                                            options=station_options,
                                            value=default_station,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="col-12 col-md-4",
                                    children=[
                                        html.Div("Estación 2", className="control-label"),
                                        dcc.Dropdown(
                                            id="compare-station-2",
                                            options=station_options,
                                            value=default_station2,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="col-12 col-md-4",
                                    children=[
                                        html.Div("Variable", className="control-label"),
                                        dcc.Dropdown(
                                            id="compare-variable",
                                            options=[
                                                {"label": VARIABLE_LABELS.get(v, v), "value": v}
                                                for v in variable_options
                                            ],
                                            value=default_variable,
                                            clearable=False
                                        ),
                                    ],
                                ),
                            ],
                        ),

                        dcc.Graph(id="compare-graph", config={"displayModeBar": False}),
                    ],
                )
            ],
        ),

        # Estado oculto para guardar la estación seleccionada en el mapa
        dcc.Store(id="selected-station-store", data=default_station),
    ],
)



# -----------------------------------------------------------
# CALLBACK: mapa + selección de estación
# -----------------------------------------------------------

@app.callback(
    Output("map-graph", "figure"),
    Output("selected-station-store", "data"),
    Output("selected-station-label", "children"),
    Input("date-picker", "date"),
    Input("map-graph", "clickData"),
    State("selected-station-store", "data"),
)
def update_map(date_value, click_data, current_station_id):
    if date_value is None:
        return go.Figure(), current_station_id, "Ninguna estación seleccionada"

    date_value = pd.to_datetime(date_value)

    # 1) Determinar nueva estación seleccionada (solo si el click tiene customdata)
    selected_station_id = current_station_id
    if click_data and "points" in click_data and click_data["points"]:
        cd = click_data["points"][0].get("customdata")
        if cd is not None:
            selected_station_id = str(cd)

    # 2) Filtrar datos para la fecha
    dff = df[df["date"] == date_value].copy()
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Sin datos para la fecha {date_value.date()}",
            mapbox_style="open-street-map",
            mapbox_center={"lat": center_lat, "lon": center_lon},
            mapbox_zoom=10,
        )
        return fig, selected_station_id, "Ninguna estación seleccionada"

    # 3) Crear figura desde cero
    fig = go.Figure()

    # 3a) Capa de fondo: borde negro para todas las estaciones
    fig.add_trace(
        go.Scattermapbox(
            lat=dff["lat"],
            lon=dff["lon"],
            mode="markers",
            marker=dict(
                size=22,
                color="black",
                opacity=1.0,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 3b) Capa principal: círculos coloreados por temperatura (con customdata para el click)
    fig.add_trace(
        go.Scattermapbox(
            lat=dff["lat"],
            lon=dff["lon"],
            mode="markers",
            marker=dict(
                size=16,
                color=dff[TEMPERATURE_COL],
                colorscale="RdYlBu_r",
                cmin=temp_min,
                cmax=temp_max,
                colorbar=dict(title="Temp (°C)"),
            ),
            customdata=dff["station_id"],
            text=dff["station_name"],
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Temp: %{marker.color:.2f} °C<br>" +
                "Estación: %{customdata}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # 4) Capa de resaltado: estación seleccionada en rojo
    sel = dff[dff["station_id"] == str(selected_station_id)]
    if not sel.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=sel["lat"],
                lon=sel["lon"],
                mode="markers",
                marker=dict(
                    size=26,
                    color="red",
                    opacity=0.9,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        station_name = sel["station_name"].iloc[0]
        selected_label = f"Estación seleccionada: {station_name} ({selected_station_id})"
    else:
        selected_label = "Ninguna estación seleccionada"

    # 5) Layout del mapa
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=10,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    return fig, selected_station_id, selected_label




# -----------------------------------------------------------
# CALLBACK: panel de datos del día
# -----------------------------------------------------------

@app.callback(
    Output("station-data-panel", "children"),
    Input("date-picker", "date"),
    Input("selected-station-store", "data"),
)
def update_station_panel(date_value, station_id):
    if date_value is None or station_id is None:
        return "Selecciona una fecha y una estación."

    date_value = pd.to_datetime(date_value)
    station_id = str(station_id)

    row = df[(df["date"] == date_value) & (df["station_id"] == station_id)]
    if row.empty:
        return "No hay datos para esta estación en la fecha seleccionada."

    row = row.iloc[0]

    is_pred = bool(row.get("is_pred", False))
    tipo_str = "Predicción" if is_pred else "Dato observado"

    # ---------- Mapeo nombres + unidades ----------
    campos = [
        ("temperature", "Temperatura", "°C"),
        ("relative_humidity", "Humedad relativa", "%"),
        ("precipitation", "Precipitación", "mm"),
        ("pressure", "Presión", "hPa"),
        ("solar_radiation", "Radiación solar", "W/m²"),
        ("wind_speed", "Velocidad del viento", "m/s"),
        ("wind_direction", "Dirección del viento", "°"),
        ("altitude", "Altitud", "m"),
    ]

    # ---------- Emoji / icono de tiempo ----------
    temp_value = row.get("temperature", None)
    precip_value = row.get("precipitation", 0.0)

    weather_emoji = "❓"
    if pd.notna(temp_value):
        t = float(temp_value)
        p = float(precip_value) if pd.notna(precip_value) else 0.0

        if p >= 5:
            weather_emoji = "🌧️"   # lluvia fuerte
        elif p > 0:
            weather_emoji = "🌦️"   # chubascos
        else:
            if t < 0:
                weather_emoji = "❄️"
            elif t < 8:
                weather_emoji = "🥶"
            elif t < 16:
                weather_emoji = "🌥️"
            elif t < 24:
                weather_emoji = "🌤️"
            elif t < 30:
                weather_emoji = "☀️"
            else:
                weather_emoji = "🥵"

    # Métrica principal: temperatura + emoji
    big_metric = None
    if pd.notna(temp_value):
        big_metric = html.Div(
            className="station-big-metric",
            children=[
                html.Div("Temperatura", className="station-big-label"),
                html.Div(
                    f"{temp_value:.1f} °C {weather_emoji}",
                    className="station-big-value",
                ),
            ],
        )

    # ---------- Tarjetas pequeñas para el resto ----------
    metric_cards = []
    for key, label, unit in campos:
        if key == "temperature":
            continue  # ya usada como métrica grande
        if key not in row or pd.isna(row[key]):
            continue
        val = row[key]
        if isinstance(val, (float, int)):
            val_str = f"{val:.1f}" if abs(val) < 1000 else f"{val:.0f}"
        else:
            val_str = str(val)

        metric_cards.append(
            html.Div(
                className="station-metric-card",
                children=[
                    html.Div(label, className="station-metric-label"),
                    html.Div(f"{val_str} {unit}", className="station-metric-value"),
                ],
            )
        )

    # ---------- Encabezado ----------
    header = html.Div(
        className="station-header",
        children=[
            html.Div(
                children=[
                    html.Div(
                        f"{row['station_name']} ({station_id})",
                        className="station-name",
                    ),
                    html.Div(
                        f"{date_value.date()}",
                        className="station-date",
                    ),
                ]
            ),
            html.Span(
                tipo_str,
                className=(
                    "station-badge station-badge-pred"
                    if is_pred
                    else "station-badge station-badge-real"
                ),
            ),
        ],
    )

    return html.Div(
        children=[
            header,
            big_metric if big_metric is not None else None,
            html.Div(
                className="station-metric-grid",
                children=metric_cards,
            ),
        ]
    )




# -----------------------------------------------------------
# CALLBACK: gráfico de comparación de dos estaciones
# -----------------------------------------------------------

@app.callback(
    Output("compare-graph", "figure"),
    Input("compare-station-1", "value"),
    Input("compare-station-2", "value"),
    Input("compare-variable", "value"),
)
def update_compare_graph(station1, station2, variable):
    if not station1 or not station2 or not variable:
        return go.Figure()

    station1 = str(station1)
    station2 = str(station2)

    dff = df[df["station_id"].isin([station1, station2])].copy()
    if dff.empty:
        return go.Figure()

    # Solo filas con valor en esa variable
    dff = dff[dff[variable].notna()].copy()
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Sin datos para la variable '{variable}' en estas estaciones",
            xaxis_title="Fecha",
            yaxis_title=variable,
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            font=dict(color="#e5e7eb"),
        )
        return fig

    dff = dff.sort_values("date")

    # Nombre "bonito" de la variable
    variable_nombres = {
        "temperature": "Temperatura (°C)",
        "precipitation": "Precipitación (mm)",
        "pressure": "Presión (hPa)",
        "relative_humidity": "Humedad relativa (%)",
        "solar_radiation": "Radiación solar (W/m²)",
        "wind_speed": "Velocidad viento (m/s)",
        "wind_direction": "Dirección viento (°)",
    }
    var_label = VARIABLE_LABELS.get(variable, variable)

    # Tipo de dato (real / predicción)
    dff["tipo"] = dff["is_pred"].map({False: "Real", True: "Predicción"})

    # Colores por estación (2 estaciones máximo)
    station_names = dff["station_name"].unique().tolist()
    color_sequence = ["#60a5fa", "#f97316", "#22c55e", "#e11d48"]
    fig = px.line(
        dff,
        x="date",
        y=variable,
        color="station_name",
        line_dash="tipo",
        color_discrete_sequence=color_sequence[: len(station_names)],
        labels={
            "station_name": "Estación",
            "tipo": "Tipo",
            "date": "Fecha",
            variable: var_label,
        },
        title=f"Evolución de {var_label} en dos estaciones",
    )

    # Hacer las líneas un poco más finas
    fig.update_traces(line=dict(width=1.4))

    # Línea vertical en inicio de predicciones (primera fecha con is_pred=True)
    cutoff = dff.loc[dff["is_pred"] == True, "date"].min()
    if pd.notna(cutoff):
        y_min = dff[variable].min()
        y_max = dff[variable].max()

        fig.add_shape(
            type="line",
            x0=cutoff,
            x1=cutoff,
            y0=y_min,
            y1=y_max,
            xref="x",
            yref="y",
            line=dict(color="#9ca3af", dash="dash", width=2),
        )

        fig.add_annotation(
            x=cutoff,
            y=y_max,
            text="Inicio predicciones",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11, color="#9ca3af"),
        )

    # Estilo oscuro a juego con la app
    fig.update_layout(
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(
            title="Fecha",
            gridcolor="#374151",
            zerolinecolor="#4b5563",
        ),
        yaxis=dict(
            title=var_label,
            gridcolor="#374151",
            zerolinecolor="#4b5563",
        ),
        legend=dict(
            title="Estación / tipo de dato",
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="#4b5563",
            borderwidth=1,
        ),
    )

    return fig

@app.callback(
    Output("nearby-fields-panel", "children"),
    Input("date-picker", "date"),
    Input("selected-station-store", "data"),
)
def update_nearby_fields(date_value, station_id):
    # Solo necesitamos la estación; la fecha aquí no importa
    if station_id is None:
        return "Selecciona una estación en el mapa."

    station_id = str(station_id)

    # Tomamos una fila cualquiera de esa estación (lat/lon fijos por estación)
    station_row = df[df["station_id"] == station_id].dropna(subset=["lat", "lon"])
    if station_row.empty:
        return "No hay coordenadas para esta estación."

    station_row = station_row.iloc[0]
    s_lat = station_row["lat"]
    s_lon = station_row["lon"]

    # Calculamos distancias a cada campo de fútbol
    def compute_dist(row):
        return haversine_distance(s_lat, s_lon, float(row["lat"].replace(",", ".")), float(row["lon"].replace(",", ".")))

    tmp = fields_df.copy()
    tmp["dist_m"] = tmp.apply(compute_dist, axis=1)

    # Filtrar campos a ≤ 500 m
    nearby = tmp[tmp["dist_m"] <= 1000].sort_values("dist_m")

    if nearby.empty:
        return html.Div(
            "No hay campos de fútbol municipales a menos de 1km.",
            style={"color": "#9ca3af"},
        )

    items = []
    for _, r in nearby.iterrows():
        # Construir dirección: street_type + street + number
        street_type = r.get("street_type", "")
        street = r.get("street", "")
        number = r.get("number", "")

        parts = []
        if isinstance(street_type, str) and street_type.strip():
            parts.append(street_type.strip())
        if isinstance(street, str) and street.strip():
            parts.append(street.strip())

        main_street = " ".join(parts)
        # Normalización del número (evitar 107.0 → 107)
        def clean_number(x):
            if pd.isna(x):
                return ""
            if isinstance(x, str):
                return x.strip()
            try:
                # Si es un número flotante como 107.0 → 107
                if float(x).is_integer():
                    return str(int(float(x)))
                else:
                    return str(x)
            except:
                return str(x)

        num_str = clean_number(number)

        if main_street and num_str:
            address = f"{main_street} {num_str}"
        elif main_street:
            address = main_street
        else:
            address = "Dirección no disponible"

        # Teléfono
        phone = r.get("phone", "")
        if isinstance(phone, str) and phone.strip():
            phone_text = f"📞 {phone.strip()}"
        else:
            phone_text = "📞 No disponible"

        # Email
        email = r.get("email", "")
        if isinstance(email, str) and email.strip():
            email_text = f"✉️ {email.strip()}"
        else:
            email_text = "✉️ No disponible"

        items.append(
            html.Div(
                className="field-item",
                children=[
                    html.Div("⚽ " + r["field_name"], className="field-name"),
                    html.Div(address, className="field-meta"),
                    html.Div(phone_text, className="field-meta"),
                    html.Div(email_text, className="field-meta"),
                    html.Div(
                        f"Distancia: {r['dist_m']:.0f} m",
                        className="field-meta",
                    ),
                ],
            )
        )

    return html.Div(items)



# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
