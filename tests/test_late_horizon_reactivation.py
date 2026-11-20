from rdmatcher import MatchingOptions, ExecutionOptions
import numpy as np
import pandas as pd
from rdmatcher import RDMatcher

def test_late_safe_loss_expands_beyond_cap_number_of_rounds():
    """A cap of 10 may require more than ten synchronized rounds."""
    shared = np.linspace(0.0, 0.01, 9)
    cases = np.concatenate([[1.0], shared - 0.02])
    controls = np.concatenate([shared, [2.2]])
    df = pd.DataFrame({'patient_id': np.arange(20), 'exposure': [1] * 10 + [0] * 10, 'x': np.concatenate([cases, controls])})
    model = RDMatcher(df, 'patient_id', 'exposure', ['x'], [], process_features=False, onehot=False, log_file=None, log_to_console=False)
    graph = model.return_graph(threshold=np.inf, n_neighbors=1, distance_metric='euclidean', execution=ExecutionOptions(n_jobs=1))
    focal = int(np.where(graph.case_ids == 0)[0][0])
    private = int(np.where(graph.control_ids == 19)[0][0])
    assert graph.candidate_horizons[focal] == 10
    assert graph.matrix[focal, private] > 0
    matched = model.rare_matching(threshold=np.inf, n_neighbors=1, distance_metric='euclidean', execution=ExecutionOptions(solver='scipy_sparse', n_jobs=1))
    assert model.assignment_diagnostics_['preservation_condition_satisfied']
    total = 0.0
    pairs = 0
    for _, group in matched.groupby('match_group'):
        a = group[group.exposure.eq(1)]
        b = group[group.exposure.eq(0)]
        if len(a) and len(b):
            total += abs(float(a.iloc[0].x) - float(b.iloc[0].x))
            pairs += 1
    assert pairs == 10
    assert abs(total - 1.38) < 1e-06
