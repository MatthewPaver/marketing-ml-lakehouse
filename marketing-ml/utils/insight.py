from __future__ import annotations

import re
import streamlit as st


def so_what(title: str, observation: str, why: str, action: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(f"- Observation: {observation}")
        st.markdown(f"- Why it matters: {why}")
        st.markdown(f"- Action: {action}")


def bullets_to_md(actions: list[str]) -> str:
    return "\n".join(f"- {a}" for a in actions)


def sanitize_lm_text(text: str) -> str:
    """Normalise LM output to clean single-line bullets with GBP and compact spacing."""
    if not text:
        return "- No suggestions"
    # Replace dollar with pound
    t = text.replace("$", "£")
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # Split into candidate bullets by dash markers
    parts = re.split(r"\s*[-•]\s+", t)
    bullets = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Keep within reasonable length
        bullets.append(p[:300])
        if len(bullets) == 3:
            break
    if not bullets:
        bullets = [t[:300]]
    return "\n".join(f"- {b}" for b in bullets)
