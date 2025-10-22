# AIMS Health Risk Analytics

This repository contains an end-to-end analytical workflow used by the AIMS team to explore community health screening data and surface actionable insights. The core deliverable is the `nurul_bashar.ipynb` notebook, which walks through data ingestion, cleansing, feature engineering, predictive modelling, and visual reporting.

## Repository contents
- `nurul_bashar.ipynb` — interactive analysis covering data preparation, risk scoring, model evaluation, clustering, and dashboard-ready visuals.
- `data/` — input and derived datasets used by the notebook:
  - `test-dataset.xlsx - test data.csv` (raw screening records)
  - `dataset_variable_description.xlsx - Sheet1.csv` (field glossary)
  - `processed_health_data.csv` (export produced in notebook Cell 35)
- `nurul_bashar_technical_report.pdf` — companion report summarising findings.
- `requirements.txt` — Python dependencies required to execute the notebook.
- `LICENSE` — project licensing information.

## Getting started
1. **Set up Python**  
   Create and activate a virtual environment (example shown with PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**  
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Running the analysis
1. Launch Jupyter Lab or VS Code and open `nurul_bashar.ipynb`.
2. Ensure the kernel points to the `.venv` you created.
3. Execute the notebook top-to-bottom. The pipeline will:
   - Standardise raw screening data and engineer risk-related features.
   - Train a class-weighted Random Forest model and evaluate performance.
   - Generate visual diagnostics (risk distributions, confusion matrix, feature importances).
   - Segment households via clustering and export `data/processed_health_data.csv`.
   - Provide a methodology timeline summarising the project flow.

If you only need the processed output, you can re-run Cell 35 to refresh `processed_health_data.csv` after any modifications.

## Reproducing figures and reports
All Plotly figures render inline inside the notebook. To export static images, install the optional `kaleido` dependency (already listed in `requirements.txt`) and call `fig.write_image(...)` on the relevant cells. The PDF report can be regenerated manually by re-running the notebook and exporting the results as needed.

## Support
For clarifications or enhancements, open an issue or contact the project maintainer within the AIMS team.
