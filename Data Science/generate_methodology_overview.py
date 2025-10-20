from __future__ import annotations

import io
from datetime import datetime

import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from health_risk_pipeline import MODEL_DECISION_THRESHOLD

OUTPUT_PATH = "nurul_bashar_methodology_overview.pdf"


def build_methodology_diagram() -> go.Figure:
    steps = [
        ("Raw Data Intake", "CSV ingestion & schema validation"),
        ("Cleaning & Standardisation", "Canonical labels, duplicate removal, missing tags"),
        ("Feature Engineering", "Risk scoring, chronic signals, socioeconomic encoding"),
        (
            "Model Training & Validation",
            f"Imputation + scaling + class-weighted RF\nThreshold = {MODEL_DECISION_THRESHOLD:.2f}",
        ),
        ("Segmentation & Reporting", "Clustering, hotspot ranking, deliverables"),
    ]

    width = 0.17
    gap = 0.03
    shapes = []
    annotations = []
    arrows = []

    for idx, (title, subtitle) in enumerate(steps):
        x0 = idx * (width + gap)
        x1 = x0 + width
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                y0=0.35,
                x1=x1,
                y1=0.75,
                line=dict(color="#1f77b4", width=2),
                fillcolor="rgba(31, 119, 180, 0.12)",
                layer="below",
            )
        )
        annotations.extend(
            [
                dict(
                    x=(x0 + x1) / 2,
                    y=0.63,
                    text=f"<b>{title}</b>",
                    showarrow=False,
                    font=dict(size=14),
                ),
                dict(
                    x=(x0 + x1) / 2,
                    y=0.46,
                    text=subtitle,
                    showarrow=False,
                    font=dict(size=11),
                ),
            ]
        )

        if idx < len(steps) - 1:
            arrows.append(
                dict(
                    ax=x1,
                    ay=0.55,
                    axref="x",
                    ayref="y",
                    x=x1 + gap - 0.01,
                    y=0.55,
                    xref="x",
                    yref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#444",
                )
            )

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations + arrows,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=360,
        width=1200,
        title="End-to-End Methodology",
        margin=dict(l=40, r=40, t=80, b=40),
        template="plotly_white",
    )
    return fig


def create_pdf(path: str = OUTPUT_PATH) -> None:
    fig = build_methodology_diagram()
    png_bytes = fig.to_image(format="png", scale=3)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "End-to-End Methodology Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "This diagram distills the complete workflow used in the health risk assessment project - from raw data "
        "ingestion through feature engineering, model calibration, and downstream segmentation/reporting. "
        "Each stage is designed for reproducibility and clinical interpretability.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf.ln(4)
    pdf.image(io.BytesIO(png_bytes), x=15, y=pdf.get_y(), w=180)

    pdf.set_y(250)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        f"Generated on {datetime.now():%d %B %Y}. See nurul_bashar_technical_report.pdf for detailed narrative "
        "and the nurul_bashar.ipynb notebook for executable analysis.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    pdf.output(path)


def main() -> None:
    create_pdf()
    print(f"Methodology overview saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
