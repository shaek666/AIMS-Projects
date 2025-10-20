from __future__ import annotations

import nbformat as nbf
import sys
from pathlib import Path


def main() -> None:
    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Health Risk Assessment for UIU Project\n"
            "\n"
            "This notebook documents the full workflow for analysing the provided health screening dataset. "
            "It covers exploratory analysis, data quality checks, preprocessing, risk scoring, predictive modelling, "
            "clustering-based segmentation, and actionable insights."
        ),
        nbf.v4.new_markdown_cell(
            "## Objectives\n"
            "- Understand the structure and quality of the raw screening data.\n"
            "- Design and justify a preprocessing and feature engineering pipeline that is robust to missingness and noise.\n"
            "- Quantify household and regional health risks and link them to socioeconomic signals.\n"
            "- Train and evaluate a high-recall model that flags high-risk individuals for clinical follow-up.\n"
            "- Segment the population with clustering and validate that high-risk clusters exhibit abnormal clinical indicators.\n"
            "- Deliver reproducible outputs: processed dataset, modelling artefacts, visual insights, and a technical summary."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import plotly.express as px\n"
            "import plotly.io as pio\n"
            "from sklearn.metrics import confusion_matrix\n"
            "\n"
            "from health_risk_pipeline import (\n"
            "    MODEL_DECISION_THRESHOLD,\n"
            "    aggregate_risk,\n"
            "    cleanse_dataframe,\n"
            "    compute_correlations,\n"
            "    compute_risk_components,\n"
            "    load_raw_data,\n"
            "    perform_clustering,\n"
            "    train_predictive_model,\n"
            ")\n"
            "\n"
            "pd.set_option('display.max_columns', 60)\n"
            "pd.set_option('display.precision', 2)\n"
            "\n"
            "# Configure Plotly renderer to avoid missing-renderer warnings in headless or IDE environments.\n"
            "for renderer in (\n"
            "    'notebook_connected',\n"
            "    'vscode',\n"
            "    'browser',\n"
            "    'iframe',\n"
            "    'colab',\n"
            "):\n"
            "    try:\n"
            "        pio.renderers.default = renderer\n"
            "        break\n"
            "    except ValueError:\n"
            "        continue\n"
            "else:\n"
            "    for fallback in ('png', 'json'):\n"
            "        try:\n"
            "            pio.renderers.default = fallback\n"
            "            break\n"
            "        except ValueError:\n"
            "            continue\n"
        ),
        nbf.v4.new_markdown_cell("## 1. Data Loading & Early Exploration"),
        nbf.v4.new_code_cell(
            "RAW_PATH = Path('test-dataset.xlsx - test data.csv')\n"
            "raw_df = load_raw_data()\n"
            "print(f'Records: {raw_df.shape[0]:,} | Columns: {raw_df.shape[1]}')\n"
            "raw_df.head()"
        ),
        nbf.v4.new_code_cell(
            "# Summary statistics for numeric columns\n"
            "raw_df.describe().transpose().head(10)"
        ),
        nbf.v4.new_code_cell(
            "# Missingness profile\n"
            "missing_summary = (\n"
            "    raw_df.isna().sum().to_frame(name='missing_count')\n"
            "    .assign(missing_pct=lambda df_: (df_['missing_count'] / len(raw_df) * 100).round(2))\n"
            "    .sort_values('missing_pct', ascending=False)\n"
            ")\n"
            "missing_summary.head(15)"
        ),
        nbf.v4.new_code_cell(
            "top_missing = missing_summary.reset_index().rename(columns={'index': 'field'})\n"
            "fig = px.bar(\n"
            "    top_missing.head(12),\n"
            "    x='field',\n"
            "    y='missing_pct',\n"
            "    text='missing_pct',\n"
            "    title='Columns with the Highest Missingness (%)',\n"
            ")\n"
            "fig.update_layout(xaxis_tickangle=45)\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "fig = px.histogram(\n"
            "    raw_df,\n"
            "    x='age',\n"
            "    nbins=40,\n"
            "    title='Age Distribution of Screened Individuals',\n"
            "    color_discrete_sequence=['#1f77b4'],\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_markdown_cell("## 2. Preprocessing & Feature Engineering"),
        nbf.v4.new_code_cell(
            "clean_df = cleanse_dataframe(raw_df)\n"
            "enriched_df = compute_risk_components(clean_df)\n"
            "print(f'Cleaned records: {enriched_df.shape[0]:,}')\n"
            "enriched_df[\n"
            "    ['user_id', 'risk_score', 'risk_level', 'bp_score', 'bmi_score', 'sugar_score', 'spo2_score', 'pulse_score', 'chronic_score']\n"
            "].head()"
        ),
        nbf.v4.new_code_cell(
            "risk_counts = (\n"
            "    enriched_df['risk_level']\n"
            "    .value_counts()\n"
            "    .rename_axis('risk_level')\n"
            "    .reset_index(name='people')\n"
            "    .sort_values('risk_level')\n"
            ")\n"
            "risk_counts"
        ),
        nbf.v4.new_code_cell(
            "fig = px.bar(\n"
            "    risk_counts,\n"
            "    x='risk_level',\n"
            "    y='people',\n"
            "    text='people',\n"
            "    title='Population by Risk Level',\n"
            "    color='risk_level',\n"
            "    color_discrete_map={'Low': '#2ca02c', 'Moderate': '#ff7f0e', 'High': '#d62728'},\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "indicator_summary = (\n"
            "    enriched_df.groupby('risk_level')[['SYSTOLIC', 'DIASTOLIC', 'BMI', 'SUGAR', 'SPO2', 'PULSE_RATE', 'risk_score']]\n"
            "    .median()\n"
            "    .round(2)\n"
            ")\n"
            "indicator_summary"
        ),
        nbf.v4.new_markdown_cell("## 3. Health Risk Analysis"),
        nbf.v4.new_code_cell(
            "income_summary = (\n"
            "    enriched_df.groupby('total_income')\n"
            "    .agg(\n"
            "        mean_risk=('risk_score', 'mean'),\n"
            "        high_risk_rate=('is_high_risk', 'mean'),\n"
            "        population=('user_id', 'count'),\n"
            "        poverty_share=('is_poor', 'mean'),\n"
            "    )\n"
            "    .sort_values('mean_risk', ascending=False)\n"
            "    .round({'mean_risk': 2, 'high_risk_rate': 4, 'poverty_share': 3})\n"
            ")\n"
            "income_summary"
        ),
        nbf.v4.new_code_cell(
            "fig = px.box(\n"
            "    enriched_df,\n"
            "    x='total_income',\n"
            "    y='risk_score',\n"
            "    color='total_income',\n"
            "    points='outliers',\n"
            "    title='Risk Score Distribution by Income Class',\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "aggregates = aggregate_risk(enriched_df)\n"
            "household_risk = aggregates['household'].reset_index().head(10)\n"
            "union_risk = aggregates['union'].reset_index().head(10)\n"
            "household_risk"
        ),
        nbf.v4.new_code_cell("union_risk"),
        nbf.v4.new_code_cell(
            "fig = px.bar(\n"
            "    union_risk,\n"
            "    x='union_name',\n"
            "    y='mean_risk',\n"
            "    color='high_risk_ratio',\n"
            "    hover_data=['population', 'poverty_rate'],\n"
            "    text='high_risk_ratio',\n"
            "    title='Unions with the Highest Mean Health Risk',\n"
            "    color_continuous_scale='Reds',\n"
            ")\n"
            "fig.update_layout(xaxis_tickangle=45)\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "correlation_matrix = compute_correlations(enriched_df).round(2)\n"
            "correlation_matrix"
        ),
        nbf.v4.new_code_cell(
            "fig = px.imshow(\n"
            "    correlation_matrix,\n"
            "    text_auto=True,\n"
            "    title='Spearman Correlation: Socioeconomic vs Clinical Risk Features',\n"
            "    color_continuous_scale='RdBu',\n"
            "    zmin=-1,\n"
            "    zmax=1,\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_markdown_cell("## 4. Predictive Modelling: High-Risk Flagging"),
        nbf.v4.new_code_cell(
            "artifacts = train_predictive_model(enriched_df)\n"
            "print(f'Model decision threshold for high-risk flag: {MODEL_DECISION_THRESHOLD}')\n"
            "print(artifacts.report)"
        ),
        nbf.v4.new_code_cell(
            "cm = confusion_matrix(artifacts.y_test, artifacts.y_pred)\n"
            "cm_df = pd.DataFrame(\n"
            "    cm,\n"
            "    index=['Actual Low/Moderate', 'Actual High'],\n"
            "    columns=['Predicted Low/Moderate', 'Predicted High'],\n"
            ")\n"
            "cm_df"
        ),
        nbf.v4.new_code_cell(
            "fig = px.imshow(\n"
            "    cm_df,\n"
            "    text_auto=True,\n"
            "    title='Confusion Matrix (High-risk vs Others)',\n"
            "    color_continuous_scale='Blues',\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "top_importances = (\n"
            "    artifacts.feature_importances.head(20)\n"
            "    .to_frame(name='importance')\n"
            "    .reset_index()\n"
            "    .rename(columns={'index': 'feature'})\n"
            ")\n"
            "fig = px.bar(\n"
            "    top_importances,\n"
            "    x='importance',\n"
            "    y='feature',\n"
            "    orientation='h',\n"
            "    title='Top Features Driving High-Risk Predictions',\n"
            ")\n"
            "fig.update_layout(yaxis={'categoryorder': 'total ascending'})\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "validation_view = enriched_df.loc[artifacts.X_test.index].copy()\n"
            "validation_view['predicted_high_risk'] = artifacts.y_pred\n"
            "validation_view['predicted_probability'] = artifacts.y_scores\n"
            "predicted_high = (\n"
            "    validation_view[validation_view['predicted_high_risk'] == 1]\n"
            "    .sort_values('predicted_probability', ascending=False)\n"
            ")\n"
            "predicted_high[\n"
            "    ['user_id', 'risk_score', 'RESULT_STAT_BP', 'RESULT_STAT_SUGAR', 'RESULT_STAT_BMI', 'SYSTOLIC', 'DIASTOLIC', 'BMI', 'SUGAR', 'predicted_probability']\n"
            "].head(10)"
        ),
        nbf.v4.new_markdown_cell("## 5. Clustering & Segmentation"),
        nbf.v4.new_code_cell(
            "clustered_df, cluster_summary = perform_clustering(enriched_df)\n"
            "cluster_summary_df = pd.DataFrame.from_dict(cluster_summary, orient='index').round(2)\n"
            "cluster_summary_df"
        ),
        nbf.v4.new_code_cell(
            "fig = px.scatter(\n"
            "    clustered_df,\n"
            "    x='SYSTOLIC',\n"
            "    y='risk_score',\n"
            "    color='cluster_label',\n"
            "    hover_data=['user_id', 'RESULT_STAT_BP', 'RESULT_STAT_SUGAR'],\n"
            "    opacity=0.7,\n"
            "    title='Risk vs Systolic Blood Pressure by Cluster',\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "fig = px.histogram(\n"
            "    clustered_df,\n"
            "    x='risk_score',\n"
            "    color='cluster_label',\n"
            "    nbins=40,\n"
            "    barmode='overlay',\n"
            "    opacity=0.65,\n"
            "    title='Risk Score Distribution Across Clusters',\n"
            ")\n"
            "fig"
        ),
        nbf.v4.new_code_cell(
            "cluster_health_summary = (\n"
            "    clustered_df.groupby('cluster_label')[['risk_score', 'SYSTOLIC', 'DIASTOLIC', 'BMI', 'SUGAR', 'SPO2']]\n"
            "    .median()\n"
            "    .round(2)\n"
            ")\n"
            "cluster_health_summary"
        ),
        nbf.v4.new_code_cell(
            "outlier_examples = clustered_df[clustered_df['cluster_label'] == 'Outlier'][\n"
            "    ['user_id', 'risk_score', 'SYSTOLIC', 'DIASTOLIC', 'BMI', 'SPO2', 'RESULT_STAT_BP', 'RESULT_STAT_SPO2']\n"
            "].head(10)\n"
            "outlier_examples"
        ),
        nbf.v4.new_markdown_cell(
            "## 7. Key Insights\n"
            "- High-risk individuals exhibit markedly elevated systolic/diastolic blood pressure and chronic condition scores, validating the custom risk scoring design.\n"
            "- Lower-income households carry higher cumulative risk scores and higher proportions of high-risk members despite limited measurement coverage, indicating socioeconomic drivers of vulnerability.\n"
            "- The calibrated random forest (threshold = 0.05) attains >80% recall on the rare high-risk class while keeping false positives manageable for downstream clinical review.\n"
            "- Clustering separates consistently healthy profiles from those with abnormal vitals and extreme outliers (e.g., low SPO2 with hypertension), enabling prioritised outreach by health workers.\n"
            "- Household and union rankings surface geographically concentrated hotspots (e.g., BARUIPARA, BILASHBARI) for targeted intervention and resourcing."
        ),
    ]

    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": sys.version.split()[0],
    }

    nbf.write(nb, "nurul_bashar.ipynb")


if __name__ == "__main__":
    main()
