from __future__ import annotations

import io
import streamlit as st

try:
    import pdfkit  # type: ignore
except Exception:  # pragma: no cover
    pdfkit = None


def export_csv(df, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=f"Download {label} CSV", data=csv, file_name=f"{label}.csv", mime="text/csv")


def export_pdf(html: str, filename: str) -> None:
    if pdfkit is None:
        st.download_button(label=f"Download {filename}.html", data=html, file_name=f"{filename}.html", mime="text/html")
        return
    try:
        pdf_bytes = pdfkit.from_string(html, False)
        st.download_button(label=f"Download {filename}.pdf", data=pdf_bytes, file_name=f"{filename}.pdf", mime="application/pdf")
    except Exception:
        st.download_button(label=f"Download {filename}.html", data=html, file_name=f"{filename}.html", mime="text/html")
