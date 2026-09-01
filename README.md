# Pet Project

A Flask web app that predicts pet insurance risk, medical costs, and claims probability using trained ML models.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Files

- `app.py` — Flask application / API
- `train_model.py`, `train_claims_models.py`, `generate_data.py` — model training scripts
- `*.pkl` — trained models (tracked via Git LFS: `cost_model`, `risk_model`, `claims_probability_model`, `medical_cost_model`, plus `encoders.pkl`)
- `*.csv` — training/reference datasets
- `templates/` — HTML templates for the web UI

## Note on large files

`claims_probability_model.pkl` (~300MB) and `medical_cost_model.pkl` (~250MB) are **not** included in this repo (see `.gitignore`) — they're too large to push without Git LFS. To run the app, place your own copies of these two files in the project root before starting `app.py`.
