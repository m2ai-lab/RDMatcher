import warnings
import numpy as np
import pandas as pd

from rdmatcher.train import propensity_logits_simple


def test_propensity_fit_emits_no_sklearn_penalty_warning():
    # Small deterministic dataset
    df = pd.DataFrame({
        'exposure': [1, 0, 0, 1, 0, 1],
        'age': [20, 30, 40, 50, 60, 70],
        'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    })

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _df, logits, meta = propensity_logits_simple(df.copy(), exposure_status='exposure', all_features=['age', 'x'])

        # Ensure no FutureWarning about 'penalty' from sklearn appears
        msgs = [str(x.message) for x in w]
        assert not any("'penalty' was deprecated" in m or "penalty' was deprecated" in m for m in msgs)

    # Basic sanity: logits column present
    assert 'propensity_logit' in _df.columns
    assert len(logits) == len(df)
