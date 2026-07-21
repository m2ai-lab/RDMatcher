from __future__ import annotations

RESEARCH_CONFIG: dict[str, dict[str, str]] = {
    "PSM": {"name": "PSM (RDM)", "color": "#B7A6CA"},
    "RDM": {"name": "RDM", "color": "#314B5C"},
    "PSM_RDM": {"name": "PSM+RDM", "color": "#7698A6"},
    "RDM_Mahalanobis": {"name": "Maha (RDM)", "color": "#405936"},
    "PSM_RDM_Mahalanobis": {"name": "PSM+Maha (RDM)", "color": "#819E80"},
    "MatchIt": {"name": "PSM (MatchIt)", "color": "#DE9E48"},
    "MatchIt_ScaledEuclidean": {"name": "Scaled Euclidean (MatchIt)", "color": "#B85B3F"},
    "Mahalanobis": {"name": "Maha (MatchIt)", "color": "#732E34"},
    "Hybrid_Maha_MatchIt": {"name": "PSM+Maha (MatchIt)", "color": "#C47C8C"},
}

METHOD_ORDER: list[str] = [
    "PSM", "RDM", "PSM_RDM", "RDM_Mahalanobis",
    "PSM_RDM_Mahalanobis", "MatchIt", "MatchIt_ScaledEuclidean",
    "Mahalanobis", "Hybrid_Maha_MatchIt",
]

METHOD_LABELS = {method: RESEARCH_CONFIG[method]["name"] for method in METHOD_ORDER}
PALETTE = {method: RESEARCH_CONFIG[method]["color"] for method in METHOD_ORDER}
LABEL_ORDER = [METHOD_LABELS[method] for method in METHOD_ORDER]
LABEL_PALETTE = {METHOD_LABELS[method]: PALETTE[method] for method in METHOD_ORDER}
