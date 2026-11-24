# Demand Forecasting

Predict simulated weekly demand using lag features and compare against a naive previous-week baseline. Chronological split; one-step forecasts use observed previous values. Synthetic data only.

```bash
pip install -r requirements.txt
python forecast.py
python -m pytest -q
```
The local reference run reported model MAE 11.55 and baseline MAE 9.51. The baseline was better; the complex model should not be deployed based on this result. Running the script generates predictions.csv and forecast.png.

Independent AI-assisted demonstration.
