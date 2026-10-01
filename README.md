# Predictive Maintenance: Remaining Useful Life Prediction

Uncertainty-aware, explainable, cost-optimal RUL prediction on NASA C-MAPSS.
See [plan.md](plan.md) for the full semester plan and [reports/scope.md](reports/scope.md) for the scope.

## Setup
```bash
pip install -r requirements.txt
```
The raw data lives in `data/raw/` (not committed). To re-download it:
```bash
curl -L -o cmapss.zip "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
```
Unzip it, then unzip the inner `CMAPSSData.zip` into `data/raw/`.

## Verify the data
```bash
python -m src.check_data
```

## Layout
```
data/raw/         C-MAPSS text files
data/processed/   cleaned / windowed data
notebooks/        EDA and experiments
src/              reusable code (data loading, features, models)
models/           saved model weights
app/              Streamlit demo
reports/          scope, literature review, figures, final report
```
