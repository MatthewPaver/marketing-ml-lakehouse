from __future__ import annotations

import streamlit as st


def fmt_gbp(x: float | int | None) -> str:
    if x is None:
        return "£0"
    return f"£{x:,.0f}"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "0%"
    return f"{x*100:.1f}%" if x <= 1 else f"{x:.1f}%"


def kpi_tile(label: str, value: str, delta: str | None = None) -> None:
    if delta is None:
        st.metric(label, value)
    else:
        st.metric(label, value, delta)
