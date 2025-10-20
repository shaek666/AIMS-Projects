"""Reusable analysis utilities for the health risk assessment task."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_PATH = Path("test-dataset.xlsx - test data.csv")
METADATA_PATH = Path("dataset_variable_description.xlsx - Sheet1.csv")


INCOME_ORDER = [
    "Lower class",
    "Lower-middle class",
    "Middle class",
    "Upper class",
]

BP_RISK = {
    "normal": 0,
    "low": 2,
    "prehypertension": 2,
    "mild high": 3,
    "moderate high": 4,
    "high": 5,
    "severe high": 6,
}

BMI_RISK = {
    "normal": 0,
    "underweight": 3,
    "overweight": 3,
    "obesity": 5,
    "highly obesity": 6,
    "morbid obesity": 6,
}

SUGAR_RISK = {
    "normal": 0,
    "pre- diabetic": 3,
    "high (borderline)": 4,
    "diabetic (need confirmation)": 5,
    "high": 5,
    "low": 3,
    "low (hypoglycemia)": 5,
}

SPO2_RISK = {
    "normal": 0,
    "low": 4,
    "very low": 6,
}

PULSE_RISK = {
    "normal": 0,
    "low": 3,
    "high": 4,
}

MODEL_DECISION_THRESHOLD = 0.05


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_scores: np.ndarray
    report: str
    roc_auc: float
    feature_importances: pd.Series


def _standardize_status(series: pd.Series) -> pd.Series:
    """Lowercase and strip whitespace for categorical status columns."""
    standardised = series.astype("string").str.strip().str.lower()
    return standardised.where(~standardised.isna(), None).astype("object")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset."""
    df = pd.read_csv(path)
    return df


def cleanse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Initial data cleaning: canonical casing, missing markers, derived helpers."""
    data = df.copy()
    data = data.drop_duplicates(subset="user_id")

    # Normalise income ordering and create ordinal view.
    data["total_income"] = (
        data["total_income"].astype("string").str.strip().where(lambda s: ~s.isna(), None)
    ).astype("object")
    income_map = {name: idx for idx, name in enumerate(INCOME_ORDER)}
    data["income_score"] = data["total_income"].map(income_map).astype("float")

    # Guard against unseen labels.
    data["income_score"] = data["income_score"].fillna(data["income_score"].median())

    # Harmonise disability flag.
    disability_series = (
        data["disabilities_name"].astype("string").str.strip().str.lower()
    )
    data["disabilities_name"] = (
        disability_series.where(~disability_series.isna(), None).astype("object")
    )
    data["has_disability"] = disability_series.replace({"nan": "", "na": ""})
    data["has_disability"] = np.where(
        (data["has_disability"] == "") | (data["has_disability"] == "0"),
        0,
        1,
    )

    # Prepare status columns.
    for column in [
        "RESULT_STAT_BP",
        "RESULT_STAT_BMI",
        "RESULT_STAT_SUGAR",
        "RESULT_STAT_SPO2",
        "RESULT_STAT_PR",
    ]:
        data[column] = _standardize_status(data[column])

    # Ensure remaining string dtypes fallback to Python objects with None for missing entries.
    string_cols = data.select_dtypes(include="string").columns
    for column in string_cols:
        data[column] = data[column].where(~data[column].isna(), None).astype("object")

    # Normalise gender labels.
    gender_std = data["gender"].astype("string").str.strip().str.lower()
    gender_std = gender_std.replace({"male": "Male", "female": "Female"})
    data["gender"] = gender_std.where(~gender_std.isna(), None).astype("object")

    # Parse birthday for completeness metrics if needed later.
    data["birthday_parsed"] = pd.to_datetime(data["birthday"], errors="coerce")

    return data


def compute_risk_components(data: pd.DataFrame) -> pd.DataFrame:
    """Translate clinical statuses into numeric signals and create helper columns."""
    df = data.copy()

    df["bp_score"] = df["RESULT_STAT_BP"].map(BP_RISK).fillna(0)
    df["bmi_score"] = df["RESULT_STAT_BMI"].map(BMI_RISK).fillna(0)
    df["sugar_score"] = df["RESULT_STAT_SUGAR"].map(SUGAR_RISK).fillna(0)
    df["spo2_score"] = df["RESULT_STAT_SPO2"].map(SPO2_RISK).fillna(0)
    df["pulse_score"] = df["RESULT_STAT_PR"].map(PULSE_RISK).fillna(0)

    df["chronic_score"] = (
        df["had_stroke"] * 6
        + df["has_cardiovascular_disease"] * 5
        + df["diabetic"].astype(int) * 4
        + df["profile_hypertensive"].astype(int) * 4
    )

    df["chronic_condition_count"] = (
        df[["had_stroke", "has_cardiovascular_disease", "diabetic", "profile_hypertensive"]]
        .astype(int)
        .sum(axis=1)
    )

    df["age_bucket_score"] = (
        pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 75, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
            right=False,
        )
        .astype("float")
        .fillna(4)
        .astype(int)
    )

    df["income_risk_score"] = (3 - df["income_score"]).clip(lower=0)
    df["poverty_score"] = df["is_poor"].astype(int) * 2
    df["disability_score"] = df["has_disability"] * 2

    measurement_cols = ["SYSTOLIC", "BMI", "SUGAR", "SPO2", "PULSE_RATE"]
    df["measured_count"] = df[measurement_cols].notna().sum(axis=1)
    df["missing_penalty"] = (len(measurement_cols) - df["measured_count"]) * 0.6

    component_cols = [
        "bp_score",
        "bmi_score",
        "sugar_score",
        "spo2_score",
        "pulse_score",
        "chronic_score",
        "age_bucket_score",
        "income_risk_score",
        "poverty_score",
        "disability_score",
        "missing_penalty",
    ]

    df["risk_raw_score"] = df[component_cols].sum(axis=1)

    max_score = (
        max(BP_RISK.values())
        + max(BMI_RISK.values())
        + max(SUGAR_RISK.values())
        + max(SPO2_RISK.values())
        + max(PULSE_RISK.values())
        + 6 + 5 + 4 + 4  # chronic contributions
        + 4  # age bucket max label
        + 3  # income_risk max
        + 2  # poverty
        + 2  # disability
        + len(measurement_cols) * 0.6  # missing penalty
    )

    df["risk_score"] = (df["risk_raw_score"] / max_score * 100).clip(upper=100)

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-np.inf, 20, 35, np.inf],
        labels=["Low", "Moderate", "High"],
    )
    df["is_high_risk"] = (df["risk_level"] == "High").astype(int)

    return df


def aggregate_risk(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create household and union risk summaries."""
    household = (
        df.groupby("household_id")
        .agg(
            household_size=("user_id", "count"),
            mean_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            high_risk_ratio=("is_high_risk", "mean"),
        )
        .sort_values("mean_risk", ascending=False)
    )

    union = (
        df.groupby("union_name")
        .agg(
            population=("user_id", "count"),
            mean_risk=("risk_score", "mean"),
            high_risk_ratio=("is_high_risk", "mean"),
            poverty_rate=("is_poor", "mean"),
        )
        .sort_values("mean_risk", ascending=False)
    )

    income = (
        df.groupby("total_income")
        .agg(
            population=("user_id", "count"),
            mean_risk=("risk_score", "mean"),
            high_risk_ratio=("is_high_risk", "mean"),
        )
        .sort_values("mean_risk", ascending=False)
    )

    return {"household": household, "union": union, "income": income}


def _prepare_model_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select modelling features and target."""
    feature_cols = [
        "age",
        "income_score",
        "bp_score",
        "bmi_score",
        "sugar_score",
        "spo2_score",
        "pulse_score",
        "chronic_score",
        "chronic_condition_count",
        "income_risk_score",
        "poverty_score",
        "disability_score",
        "missing_penalty",
        "SYSTOLIC",
        "DIASTOLIC",
        "BMI",
        "SUGAR",
        "PULSE_RATE",
        "SPO2",
    ]

    categorical_cols = [
        "gender",
        "total_income",
        "union_name",
        "disabilities_name",
        "TAG_NAME",
        "RESULT_STAT_BP",
        "RESULT_STAT_BMI",
        "RESULT_STAT_SUGAR",
        "RESULT_STAT_PR",
        "RESULT_STAT_SPO2",
    ]

    X = df[feature_cols + categorical_cols].copy()
    X = X.replace({pd.NA: np.nan})
    y = df["is_high_risk"]

    return X, y


def train_predictive_model(df: pd.DataFrame, random_state: int = 42) -> ModelArtifacts:
    """Train the predictive model pipeline and return evaluation artefacts."""
    X, y = _prepare_model_frame(df)

    numeric_cols = [
        "age",
        "income_score",
        "bp_score",
        "bmi_score",
        "sugar_score",
        "spo2_score",
        "pulse_score",
        "chronic_score",
        "chronic_condition_count",
        "income_risk_score",
        "poverty_score",
        "disability_score",
        "missing_penalty",
        "SYSTOLIC",
        "DIASTOLIC",
        "BMI",
        "SUGAR",
        "PULSE_RATE",
        "SPO2",
    ]

    categorical_cols = [
        "gender",
        "total_income",
        "union_name",
        "disabilities_name",
        "TAG_NAME",
        "RESULT_STAT_BP",
        "RESULT_STAT_BMI",
        "RESULT_STAT_SUGAR",
        "RESULT_STAT_PR",
        "RESULT_STAT_SPO2",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=10,
        class_weight={0: 1.0, 1: 15.0},
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    # Isolation forest on training data to drop extreme outliers before fitting.
    iso_numeric = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
    isolation = IsolationForest(
        contamination=0.015,
        random_state=random_state,
        n_estimators=300,
    )
    inlier_mask = isolation.fit_predict(iso_numeric) == 1
    X_train_filtered = X_train.iloc[inlier_mask]
    y_train_filtered = y_train.iloc[inlier_mask]

    if y_train_filtered.nunique() < 2:
        X_train_filtered = X_train
        y_train_filtered = y_train

    pipeline.fit(X_train_filtered, y_train_filtered)
    model_step = pipeline.named_steps["model"]

    if hasattr(model_step, "predict_proba") and len(model_step.classes_) > 1:
        positive_index = list(model_step.classes_).index(1)
        y_scores = pipeline.predict_proba(X_test)[:, positive_index]
    elif hasattr(model_step, "predict_proba"):
        y_scores = np.zeros(len(X_test), dtype=float)
    else:
        decision = pipeline.decision_function(X_test)
        decision = (decision - decision.min()) / (decision.max() - decision.min())
        y_scores = decision

    y_pred = (y_scores >= MODEL_DECISION_THRESHOLD).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low/Moderate", "High"],
        digits=3,
    )

    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except ValueError:
        roc_auc = float("nan")

    # Extract feature importances and align with names.
    cat_names = list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_cols)
    )
    feature_names = numeric_cols + cat_names
    feature_importances = pd.Series(
        model_step.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False)

    return ModelArtifacts(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_scores=y_scores,
        report=report,
        roc_auc=roc_auc,
        feature_importances=feature_importances,
    )


def perform_clustering(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Cluster individuals and flag outliers using a hybrid KMeans + IsolationForest approach."""
    features = [
        "risk_score",
        "bp_score",
        "bmi_score",
        "sugar_score",
        "spo2_score",
        "pulse_score",
        "chronic_score",
        "age",
        "income_score",
        "poverty_score",
        "missing_penalty",
    ]

    cluster_frame = df[features].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_frame.fillna(cluster_frame.median()))

    kmeans = KMeans(n_clusters=2, n_init=20, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(scaled)

    iso = IsolationForest(
        contamination=0.02,
        random_state=random_state,
        n_estimators=500,
    )
    outlier_mask = iso.fit_predict(scaled)

    cluster_mean_risk = {}
    for cluster_id in np.unique(kmeans_labels):
        cluster_mean_risk[cluster_id] = df.loc[kmeans_labels == cluster_id, "risk_score"].mean()

    # Determine which cluster is high risk.
    high_cluster = max(cluster_mean_risk, key=cluster_mean_risk.get)
    low_cluster = min(cluster_mean_risk, key=cluster_mean_risk.get)

    final_labels = np.full(len(df), "Moderate", dtype=object)
    final_labels[kmeans_labels == high_cluster] = "High-Risk Group"
    final_labels[kmeans_labels == low_cluster] = "Low-Risk Group"

    final_labels[outlier_mask == -1] = "Outlier"

    labelled = df.copy()
    labelled["cluster_label"] = final_labels
    labelled["cluster_id"] = kmeans_labels
    labelled["outlier_flag"] = (outlier_mask == -1).astype(int)

    cluster_summary = (
        labelled.groupby("cluster_label")
        .agg(
            population=("user_id", "count"),
            mean_risk=("risk_score", "mean"),
            bp_score=("bp_score", "mean"),
            bmi_score=("bmi_score", "mean"),
            sugar_score=("sugar_score", "mean"),
            spo2_score=("spo2_score", "mean"),
        )
        .sort_values("mean_risk", ascending=False)
    )

    return labelled, cluster_summary.to_dict("index")


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyse monotonic relationships between socioeconomics and clinical risk factors."""
    corr_features = [
        "income_score",
        "risk_score",
        "bp_score",
        "bmi_score",
        "sugar_score",
        "spo2_score",
        "pulse_score",
        "age",
        "poverty_score",
    ]

    corr_df = df[corr_features].copy()

    correlation_matrix = corr_df.corr(method="spearman")
    return correlation_matrix


def build_processed_dataset() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """End-to-end routine that creates the processed dataset and metrics dictionary."""
    raw = load_raw_data()
    cleaned = cleanse_dataframe(raw)
    enriched = compute_risk_components(cleaned)

    model_artifacts = train_predictive_model(enriched)
    clustered_df, cluster_summary = perform_clustering(enriched)

    correlations = compute_correlations(clustered_df)
    aggregates = aggregate_risk(clustered_df)

    overall_summary = {
        "risk_distribution": clustered_df["risk_level"].value_counts().to_dict(),
        "high_risk_share": float(clustered_df["is_high_risk"].mean()),
        "top_households": aggregates["household"]
        .head(10)
        .reset_index()
        .to_dict("records"),
        "top_unions": aggregates["union"].head(10).reset_index().to_dict("records"),
        "income_summary": aggregates["income"].reset_index().to_dict("records"),
        "model_report": model_artifacts.report,
        "roc_auc": model_artifacts.roc_auc,
        "feature_importances": model_artifacts.feature_importances.head(20).to_dict(),
        "cluster_summary": cluster_summary,
        "correlation_matrix": correlations.to_dict(),
    }

    return clustered_df, overall_summary


def save_outputs(
    processed_df: pd.DataFrame,
    summary: Dict[str, object],
    dataset_path: Path = Path("processed_health_data.csv"),
    summary_path: Path = Path("analysis_summary.json"),
) -> None:
    """Persist the processed dataset and JSON summary."""
    processed_df.to_csv(dataset_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    processed_df, summary = build_processed_dataset()
    save_outputs(processed_df, summary)
    print("Processed dataset saved to processed_health_data.csv")
    print("Analysis summary saved to analysis_summary.json")


if __name__ == "__main__":
    main()
