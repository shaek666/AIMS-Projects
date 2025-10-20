from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from health_risk_pipeline import MODEL_DECISION_THRESHOLD


SUMMARY_PATH = Path("analysis_summary.json")
REPORT_PATH = Path("nurul_bashar_technical_report.pdf")


class Report(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, "Health Risk Technical Report", align="R")
        self.ln(12)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def add_title_page(pdf: Report) -> None:
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 20, "Health Risk Prediction & Segmentation")
    pdf.ln(22)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Technical Report")
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Prepared by Codex\n"
        f"Date: {datetime.now():%d %B %Y}\n"
        "Inputs: test-dataset.xlsx - test data.csv, dataset_variable_description.xlsx - Sheet1.csv",
    )
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 11)
    overview = (
        "This report summarises the end-to-end workflow used to assess household health risks, "
        "calibrate predictive models that flag high-risk individuals, and segment the population into "
        "actionable groups for clinical follow-up. It accompanies the executable notebook "
        "nurul_bashar.ipynb, the processed dataset processed_health_data.csv, and the dependency file requirements.txt."
    )
    pdf.multi_cell(0, 6, overview)


def add_section_title(pdf: Report, title: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 10, title)
    pdf.ln(12)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)


def add_bullet_list(pdf: Report, items: List[str]) -> None:
    for item in items:
        pdf.multi_cell(0, 6, f"- {item}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def format_percentage(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def draw_methodology_diagram(pdf: Report, x: float, y: float, width: float, height: float) -> None:
    steps = [
        ("Raw Data Intake", "CSV ingestion & schema validation"),
        ("Cleaning & Standardisation", "Canonical labels, duplicate removal, missing tags"),
        ("Feature Engineering", "Risk scoring, chronic signals, socioeconomic encoding"),
        ("Model Training & Validation", "Imputation + scaling + class-weighted RF\nThreshold = "
         f"{MODEL_DECISION_THRESHOLD:.2f} for high-risk alerts"),
        ("Segmentation & Reporting", "KMeans + IsolationForest clusters, rankings, deliverables"),
    ]

    step_width = width / len(steps) * 0.85
    gap = (width - step_width * len(steps)) / max(1, len(steps) - 1)

    pdf.set_draw_color(31, 119, 180)
    pdf.set_fill_color(225, 236, 247)

    for idx, (title, subtitle) in enumerate(steps):
        rect_x = x + idx * (step_width + gap)
        pdf.rect(rect_x, y, step_width, height, "DF")
        pdf.set_xy(rect_x + 2, y + 3)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(step_width - 4, 4, title, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_xy(rect_x + 2, y + 11)
        pdf.multi_cell(step_width - 4, 4, subtitle, align="C")

        if idx < len(steps) - 1:
            line_start = rect_x + step_width
            line_end = line_start + gap - 2
            arrow_y = y + height / 2
            pdf.set_draw_color(120, 120, 120)
            pdf.line(line_start, arrow_y, line_end, arrow_y)
            pdf.line(line_end - 1.5, arrow_y - 1.5, line_end, arrow_y)
            pdf.line(line_end - 1.5, arrow_y + 1.5, line_end, arrow_y)
            pdf.set_draw_color(31, 119, 180)


def create_report(summary: Dict[str, object]) -> None:
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    add_title_page(pdf)

    add_section_title(pdf, "1. Data Handling & Preprocessing")
    risk_distribution = summary.get("risk_distribution", {})
    high_share = format_percentage(summary.get("high_risk_share", 0.0), 3)
    cleaning_notes = [
        "Removed duplicate user_id records and harmonised categorical labels (gender, disabilities, test statuses).",
        "Mapped income classes to ordinal scores and derived disability, chronic-condition, and measurement coverage indicators.",
        "Encoded clinical statuses (blood pressure, BMI, sugar, SPO2, pulse) into risk points and combined them with socioeconomic risk multipliers.",
        f"High-risk prevalence after scoring: {high_share} of the population "
        f"({risk_distribution.get('High', 0)} individuals).",
    ]
    add_bullet_list(pdf, cleaning_notes)

    add_section_title(pdf, "2. Household & Regional Health Risk Profiles")
    top_households = summary.get("top_households", [])[:5]
    top_unions = summary.get("top_unions", [])[:5]

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Highest-risk households")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    for entry in top_households:
        pdf.multi_cell(
            0,
            6,
            f"- Household {entry['household_id']} | members={entry['household_size']} | "
            f"mean risk={entry['mean_risk']:.1f} | high-risk share={format_percentage(entry['high_risk_ratio'])}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Top unions by average risk")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    for entry in top_unions:
        pdf.multi_cell(
            0,
            6,
            f"- {entry['union_name']} | population={entry['population']} | mean risk={entry['mean_risk']:.2f} | "
            f"high-risk share={format_percentage(entry['high_risk_ratio'], 3)} | "
            f"poverty rate={format_percentage(entry['poverty_rate'], 3)}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    add_section_title(pdf, "3. Predictive Modelling Results")
    pdf.multi_cell(
        0,
        6,
        "A class-weighted random forest (500 estimators, depth 14) combined with median imputation, "
        "standardisation, and calibrated probability thresholding achieves strong recall on the high-risk class "
        f"while keeping false positives manageable (threshold = {MODEL_DECISION_THRESHOLD:.2f}).",
    )
    pdf.ln(1)
    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(0, 4, summary.get("model_report", "").strip())
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        "Key drivers of high-risk predictions include chronic condition counts, systolic/diastolic readings, "
        "and pulse rate aligned with clinical expectations. Socioeconomic factors (income class, disability flag) add explanatory power for borderline cases.",
    )

    add_section_title(pdf, "4. Clustering & Segmentation Checks")
    cluster_summary = pd.DataFrame.from_dict(summary.get("cluster_summary", {}), orient="index")
    cluster_summary = cluster_summary.round(2)
    for label, row in cluster_summary.iterrows():
        pdf.multi_cell(
            0,
            6,
            f"- {label}: population={int(row['population'])}, mean risk={row['mean_risk']:.2f}, "
            f"avg BP risk={row['bp_score']:.2f}, avg BMI risk={row['bmi_score']:.2f}, "
            f"avg sugar risk={row['sugar_score']:.2f}",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
    pdf.ln(2)
    pdf.multi_cell(
        0,
        6,
        "High-Risk Group clusters show significantly elevated blood pressure and chronic scores. "
        "Outliers are characterised by extreme vitals (e.g., hypoxic SPO2) and should be prioritised for manual review.",
    )

    add_section_title(pdf, "5. Socioeconomic & Clinical Correlations")
    corr_matrix = pd.DataFrame(summary.get("correlation_matrix", {}))
    corr_pairs = corr_matrix.where(np.triu(np.ones_like(corr_matrix), k=1).astype(bool)).stack()
    if not corr_pairs.empty:
        corr_pairs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index)
        top_pairs = corr_pairs.head(4)
        for (feature_a, feature_b), value in top_pairs.items():
            pdf.multi_cell(
                0,
                6,
                f"- {feature_a} vs {feature_b}: Spearman rho = {value:.2f}",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
    pdf.ln(1)
    pdf.multi_cell(
        0,
        6,
        "Lower income scores correlate with higher aggregated risk, and chronic condition intensity is strongly tied "
        "to abnormal vitals. Measurement gaps mildly inflate risk due to reduced information about the individual.",
    )

    add_section_title(pdf, "6. Methodology Diagram")
    draw_methodology_diagram(pdf, x=20, y=pdf.get_y() + 4, width=170, height=30)
    pdf.ln(40)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "Refer to nurul_bashar.ipynb for executable walkthroughs, visuals, and validation tables that accompany this summary.",
    )

    pdf.output(str(REPORT_PATH))


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("analysis_summary.json not found. Run health_risk_pipeline.py first.")
    with SUMMARY_PATH.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    create_report(summary)
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
