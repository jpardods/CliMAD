# app.py
import dash
from dash import html
from src import graphics, etl

app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1("¡Hola, Render!"),
    html.P("Esta es una aplicación Dash desplegada en Render.")
])

if __name__ == "__main__":
    app.run()