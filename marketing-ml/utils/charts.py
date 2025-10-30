from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


def add_roas_target(fig: go.Figure, target: float = 4.0) -> go.Figure:
    fig.add_hline(y=target, line_dash="dash", line_color="#d62728", annotation_text=f"Target {target:.1f}x")
    return fig


def add_pacing_bands(fig: go.Figure) -> go.Figure:
    fig.add_hrect(y0=0.0, y1=0.9, fillcolor="#ffe6e6", opacity=0.2, line_width=0)
    fig.add_hrect(y0=0.9, y1=1.1, fillcolor="#e6ffe6", opacity=0.2, line_width=0)
    fig.add_hrect(y0=1.1, y1=2.0, fillcolor="#fff3e6", opacity=0.2, line_width=0)
    return fig


def add_ma(df: pd.DataFrame, cols: list[str], window: int = 7) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[f"{c}_ma{window}"] = out[c].rolling(window, min_periods=1).mean()
    return out
