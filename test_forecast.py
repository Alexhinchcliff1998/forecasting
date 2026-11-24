from forecast import run
from pathlib import Path
import pandas as pd

def test_holdout():
    metrics=run();pred=pd.read_csv(Path(__file__).parent/'predictions.csv')
    assert len(pred)==52 and pred.week.is_monotonic_increasing and pred.week.is_unique
    assert metrics['model_mae']>=0 and metrics['baseline_mae']>=0
