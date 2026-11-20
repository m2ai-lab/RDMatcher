import numpy as np
import pandas as pd
from rdmatcher.matcher import Matcher

def frame(seed=1, cases=30, controls=120, tied=False):
    values = np.random.default_rng(seed).normal(size=(cases + controls, 4))
    if tied:
        values = np.round(values)
    df = pd.DataFrame(values, columns=list('abcd'))
    df['id'] = np.arange(len(df))
    df['case'] = np.arange(len(df)) < cases
    return df

def make(df, metric, mode=None, **kwargs):
    model = Matcher(df=df, patient_id='id', exposure_status='case',
                   distance_metric=metric, threshold=np.inf, n_jobs=1, **kwargs)

    if mode == 'first':
        from types import MethodType
        from reference_prefilter import _prefilter_candidates
        model._prefilter_candidates = MethodType(_prefilter_candidates, model)
    return model
