from typing import Dict, List


def build_feature_name_maps(all_features: List[str],
                            original_numeric: List[str],
                            original_categorical: List[str],
                            original_datetime: List[str]) -> (Dict[str, str], Dict[str, List[str]]):
    """Map processed feature names to original features and back."""
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
        if pcol.endswith('_days'):
            base = pcol[:-5]
            if base in original_datetime:
                ocol = base
            else:
                ocol = _strip_wrappers(pcol)
        else:
            base = _strip_wrappers(pcol)
            ocol = base

        all_original = original_numeric | original_categorical | original_datetime
        if base not in all_original:
            for cat in sorted(original_categorical, key=len, reverse=True):
                if base.startswith(f"{cat}_"):
                    ocol = cat
                    break

        processed_to_original[pcol] = ocol
        original_to_processed.setdefault(ocol, []).append(pcol)

    return processed_to_original, original_to_processed
