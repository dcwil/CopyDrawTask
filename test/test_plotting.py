#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path

def load_traces(i=10):
    pth = Path(f'./templates/Size_{i}.mat')
    return scipy.io.loadmat(pth)['new_shapes'][0]


def plot_shape(shape, show_points=False, show=False):

    # create img
    df = pd.DataFrame(shape, columns=['x', 'y'])
    df['dx'] = df.x.diff()
    df['dy'] = df.y.diff()
    df['dxma'] = df.dx.rolling(2).mean()
    df['dyma'] = df.dy.rolling(2).mean()
    df['dxs'] = 0
    df['dys'] = 0
    df['dxs'][:-1] = (df.dx - df.dxma)[1:]
    df['dys'][:-1] = (df.dy - df.dyma)[1:]
    df['dxm'] = df.x + df.dxs
    df['dym'] = df.y + df.dys

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df['dxm'],
            y=df['dym'],
            name='new',
            mode='markers+lines' if show_points else 'lines',
            marker=dict(color='red'),
            line=dict(color='blue', width=20, ),
            line_shape='spline'
        )
    )
    if show_points:
        fig.add_trace(go.Scatter(
                x=df['x'],
                y=df['y'],
                name='old',
                mode='markers',
                marker=dict(color='rgba(0, 0, 0, 0)', line=dict(width=2, color='rgba(0, 1, 0, 1)')),
            )
        )

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            visible=False
        ),
        yaxis=dict(
            showgrid=False,
            visible=False
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=1080
    )

    if show:
        fig.show()

    return fig


if __name__ == "__main__":
    shapes = load_traces(i=35)
    plot_shape(shapes[0], show_points=False, show=True)

