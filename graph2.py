import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


# Définir la fonction
def f(a, d, delta, x_tilde, b):
    term1 = (1 - b * (d - a)) * delta * (1 - delta * (d ** 2 - a * d) - x_tilde * (d - a) * (1 - delta) + a)
    term2 = (1 - b * delta * (d - a)) * (1 - delta * (d ** 2 - a * d) - x_tilde * (1 - delta) + a * d)
    return term1 + term2


# Initialiser l'application Dash
app = Dash(__name__)

# Layout de l'application
app.layout = html.Div([
    dcc.Graph(
        id='graph',
        style={'height': '80vh', 'width': '80vw'}
    ),
    html.Label('δ'),
    dcc.Slider(id='delta-slider', min=0, max=1, step=0.01, value=0.96),
    html.Label('x̃'),
    dcc.Slider(id='x_tilde-slider', min=0, max=1, step=0.1, value=0.3),
    html.Label('b'),
    dcc.Slider(id='b-slider', min=0, max=4, step=1, value=2)
])


# Callback pour mettre à jour le graphique
@app.callback(
    Output('graph', 'figure'),
    [Input('delta-slider', 'value'),
     Input('x_tilde-slider', 'value'),
     Input('b-slider', 'value')]
)
def update_graph(delta, x_tilde, b):
    a = np.linspace(0, 1, 200)
    d = np.linspace(0, 1, 200)
    a, d = np.meshgrid(a, d)
    z = f(a, d, delta, x_tilde, b)

    z_positive = np.where(z >= 0, z, np.nan)
    z_negative = np.where(z <= 0, z, np.nan)

    surface_positive = go.Surface(x=a, y=d, z=z_positive, colorscale=[[0, 'red'], [1, 'red']], showscale=False)
    surface_negative = go.Surface(x=a, y=d, z=z_negative, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False)

    layout = go.Layout(
        title='Interactive 3D Function Plot',
        scene=dict(
            xaxis_title='a',
            yaxis_title='d',
            zaxis_title='f(a, d)'
        )
    )

    fig = go.Figure(data=[surface_positive, surface_negative], layout=layout)
    return fig


# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
