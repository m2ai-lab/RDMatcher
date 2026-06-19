from typing import Dict, List


def build_feature_name_maps(all_features: List[str],
                            original_numeric: List[str],
                            original_categorical: List[str],
                            original_datetime: List[str]) -> (Dict[str, str], Dict[str, List[str]]):
    """
    Build processed->original and original->processed maps based on a list of processed
    feature names (all_features) and the lists of original feature names.

    This mirrors the logic previously embedded in RDMatcher._build_feature_name_maps.
    """
    original_numeric = set(original_numeric or [])
    original_categorical = set(original_categorical or [])
    original_datetime = set(original_datetime or [])

    def _strip_wrappers(name: str) -> str:
        out = name
        changed = True
        while changed:
            changed = False
            if out.startswith('log(') and out.endswith(')'):
                out = out[4:-1]
                changed = True
            if out.startswith('binned(') and out.endswith(')'):
                out = out[7:-1]
                changed = True
        return out

    processed_to_original = {}
    original_to_processed = {}

    for pcol in all_features:
        # Datetime conversion: <col>_days maps back to <col>
        if pcol.endswith('_days'):
            base = pcol[:-5]
            if base in original_datetime:
                ocol = base
            else:
                ocol = _strip_wrappers(pcol)
        else:
            base = _strip_wrappers(pcol)
            ocol = base

        # One-hot expansions: detect if base is "{cat}_{value}" where cat is an
        # original categorical AND base itself is NOT an original feature name.
        # The extra check prevents false matches like "drug_dose" when "drug" is categorical.
        if '_' in base:
            prefix = base.split('_', 1)[0]
            all_original = original_numeric | original_categorical | original_datetime
            if prefix in original_categorical and base not in all_original:
                ocol = prefix

        processed_to_original[pcol] = ocol
        original_to_processed.setdefault(ocol, []).append(pcol)

    return processed_to_original, original_to_processed
