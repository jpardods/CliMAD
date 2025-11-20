import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px

# -----------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------

DATA_PATH = "src/data/meteo_all_predictions.csv"
TEMPERATURE_COL = "temperature"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Lista de estaciones
station_options = (
    df[["station_id", "station_name"]]
    .drop_duplicates()
    .sort_values("station_name")
    .apply(lambda x: {"label": x["station_name"], "value": x["station_id"]}, axis=1)
    .tolist()
)

# Variables numéricas para el dropdown
exclude_cols = {
    "date",
    "station_id",
    "station_name",
    "lat",
    "lon",
    "is_pred",
}
variables = [
    c for c in df.columns if c not in exclude_cols and df[c].dtype != "object"
]

default_station = station_options[0]["value"]
default_variable = variables[0]

min_date = df["date"].min()
max_date = df["date"].max()

# Rango de colores coherente para el mapa
temp_min = df[TEMPERATURE_COL].min()
temp_max = df[TEMPERATURE_COL].max()

# -----------------------------------------------------------
# INICIALIZACIÓN APP
# -----------------------------------------------------------

app = Dash(__name__)
server = app.server  # Render necesita esto


# -----------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------

app.layout = html.Div([
    html.H1("CliMAD – Visualización meteorológica y predicciones"),

    html.Div([
        # Sidebar
        html.Div([
            html.Label("Fecha"),
            dcc.DatePickerSingle(
                id="date-picker",
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                date=min_date,
                display_format="YYYY-MM-DD"
            ),

            html.Br(), html.Br(),

            html.Label("Estación"),
            dcc.Dropdown(
                id="station-dropdown",
                options=station_options,
                value=default_station,
                clearable=False
            ),

            html.P("Haz clic en un punto del mapa para seleccionar estación.")
        ],
        style={"width": "25%", "padding": "10px"}),

        # Mapa
        html.Div([
            dcc.Graph(id="map-graph")
        ], style={"width": "75%", "display": "inline-block"})
    ], style={"display": "flex"}),

    html.Hr(),

    html.Div([
        html.H3("Datos del día seleccionado"),
        html.Div(id="station-data-table")
    ]),

    html.Hr(),

    html.Div([
        html.Label("Variable para la serie temporal"),
        dcc.Dropdown(
            id="variable-dropdown",
            options=[{"label": col, "value": col} for col in variables],
            value=default_variable,
            clearable=False,
            style={"width": "300px"}
        ),

        dcc.Graph(id="time-series-graph")
    ])
])


# -----------------------------------------------------------
# CALLBACK: MAPA + click → cambia estación
# -----------------------------------------------------------

@app.callback(
    Output("map-graph", "figure"),
    Output("station-dropdown", "value"),
    Input("date-picker", "date"),
    Input("map-graph", "clickData"),
    State("station-dropdown", "value"),
)
def update_map(date_value, click_data, current_station):
    date_value = pd.to_datetime(date_value)
    dff = df[df["date"] == date_value]

    # Ver si hay click
    new_station = current_station
    if click_data:
        try:
            new_station = click_data["points"][0]["customdata"][0]
        except:
            pass

    # Mapa como scatter lat/lon (sin Mapbox)
    fig = px.scatter(
        dff,
        x="lon",
        y="lat",
        color=TEMPERATURE_COL,
        color_continuous_scale="RdYlBu_r",
        range_color=(temp_min, temp_max),
        hover_name="station_name",
        hover_data={TEMPERATURE_COL: True, "station_id": True},
        custom_data=["station_id"],
        title=f"Temperatura el {date_value.date()}",
    )

    # Destacar estación seleccionada
    fig.add_scatter(
        x=dff.loc[dff["station_id"] == new_station, "lon"],
        y=dff.loc[dff["station_id"] == new_station, "lat"],
        mode="markers",
        marker=dict(size=18, symbol="circle-open", line=dict(width=2)),
        showlegend=False,
        hoverinfo="skip",
    )

    fig.update_layout(
        yaxis_scaleanchor="x",
        xaxis_title="Longitud",
        yaxis_title="Latitud"
    )

    return fig, new_station


# -----------------------------------------------------------
# CALLBACK: tabla de datos del día
# -----------------------------------------------------------

@app.callback(
    Output("station-data-table", "children"),
    Input("date-picker", "date"),
    Input("station-dropdown", "value"),
)
def update_table(date_value, station_id):
    date_value = pd.to_datetime(date_value)

    row = df[(df["date"] == date_value) & (df["station_id"] == station_id)]
    if row.empty:
        return "No hay datos para esta fecha."

    row = row.iloc[0].to_dict()
    rows = []

    for col, val in row.items():
        if col in ["lat", "lon"]:
            continue
        rows.append(html.Tr([html.Th(col), html.Td(str(val))]))

    return html.Table([html.Tbody(rows)], style={"width": "60%"})


# -----------------------------------------------------------
# CALLBACK: Serie temporal
# -----------------------------------------------------------

@app.callback(
    Output("time-series-graph", "figure"),
    Input("station-dropdown", "value"),
    Input("variable-dropdown", "value")
)
def update_series(station_id, variable):

    dff = df[df["station_id"] == station_id].sort_values("date")

    fig = px.line(
        dff,
        x="date",
        y=variable,
        color="is_pred",
        color_discrete_map={False: "blue", True: "red"},
        labels={"is_pred": "Tipo"},
        title=f"Evolución de {variable} en {station_id}"
    )

    # renombrar leyenda
    fig.for_each_trace(
        lambda t: t.update(name="Real" if t.name == "False" else "Predicción")
    )

    # línea vertical en inicio de predicciones
    cutoff = dff.loc[dff["is_pred"], "date"].min()
    if pd.notna(cutoff):
        fig.add_vline(x=cutoff, line_dash="dash")

    fig.update_layout(xaxis_title="Fecha", yaxis_title=variable)
    return fig


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

if __name__ == "__main__":
    app.run_server(debug=True)
