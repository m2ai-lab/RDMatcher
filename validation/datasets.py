"""Dataset loaders for RDMatcher validation benchmarks."""

import os
import time
import pandas as pd
import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# ---------------------------------------------------------------------------
# RHC (Right Heart Catheterization) – Connors et al. 1996
# ---------------------------------------------------------------------------

_RHC_URL = "https://hbiostat.org/data/repo/rhc.csv"

_RHC_NUMERIC = [
    "age", "edu", "surv2md1", "das2d3pc", "adld3p",
    "aps1", "scoma1", "wtkilo1", "temp1", "meanbp1", "resp1", "hrt1",
    "pafi1", "paco21", "ph1", "wblc1", "hema1", "sod1", "pot1",
    "crea1", "bili1", "alb1", "urin1",
]

_RHC_CATEGORICAL = [
    "sex", "race", "income", "ninsclas",
    "cat1", "cat2", "ca",
    "resp", "card", "neuro", "gastr", "renal", "meta", "hema", "seps", "trauma", "ortho",
    "cardiohx", "chfhx", "dementhx", "psychhx", "chrpulhx", "renalhx",
    "liverhx", "gibledhx", "malighx", "immunhx", "transhx", "amihx",
    "dnr1",
]


def _cached_csv(url: str, cache_name: str, max_retries: int = 3) -> pd.DataFrame:
    """Download CSV with caching and retry logic."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    for attempt in range(max_retries):
        try:
            df = pd.read_csv(url)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}")


def load_rhc() -> pd.DataFrame:
    """Download and return the RHC dataset with cleaned columns.

    Returns DataFrame with:
      - patient_id: str
      - exposure_status: int (1=RHC, 0=No RHC)
      - outcome: int (1=died within 30 days, 0=alive)
      - features_numeric: list of numeric column names
      - features_categorical: list of categorical column names
    """
    df = _cached_csv(_RHC_URL, "rhc.csv")

    # Rename for consistency
    df = df.rename(columns={"ptid": "patient_id", "swang1": "exposure_status", "dth30": "outcome"})

    # Binarise treatment
    df["exposure_status"] = (df["exposure_status"] == "RHC").astype(int)

    # Binarise outcome
    df["outcome"] = (df["outcome"] == "Yes").astype(int)

    # Keep only columns that exist
    numeric = [c for c in _RHC_NUMERIC if c in df.columns]
    categorical = [c for c in _RHC_CATEGORICAL if c in df.columns]

    keep = ["patient_id", "exposure_status", "outcome"] + numeric + categorical
    df = df[keep].copy()

    # Ensure numeric columns are numeric
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    meta = {
        "features_numeric": numeric,
        "features_categorical": categorical,
    }
    df.attrs["meta"] = meta
    return df


# ---------------------------------------------------------------------------
# LaLonde / NSW – Dehejia & Wahba 1999
# ---------------------------------------------------------------------------

_LALONDE_COLS = [
    "treatment", "age", "education", "black", "hispanic",
    "married", "nodegree", "RE75", "RE78",
]
_LALONDE_COLS_DW = [
    "treatment", "age", "education", "black", "hispanic",
    "married", "nodegree", "RE74", "RE75", "RE78",
]


def _fetch_whitespace(url: str, names: list[str]) -> pd.DataFrame:
    cache_name = url.split("/")[-1].replace(".txt", ".csv")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    for attempt in range(3):
        try:
            df = pd.read_csv(url, sep=r"\s+", header=None, names=names)
            df.to_csv(cache_path, index=False)
            return df
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"Failed to download {url}: {e}")


def load_lalonde(use_re74: bool = True, external_control: str = "psid") -> pd.DataFrame:
    """Load NSW experimental data with an external comparison group.

    Parameters
    ----------
    use_re74 : bool
        If True, load the Dehejia-Wahba version that includes RE74.
    external_control : str
        'psid' or 'cps' for the non-experimental comparison group.

    Returns DataFrame with patient_id, exposure_status, outcome, and covariates.
    """
    if use_re74:
        cols = _LALONDE_COLS_DW
        treated = _fetch_whitespace(
            "https://users.nber.org/~rdehejia/data/nswre74_treated.txt",
            cols,
        )
        control_nsw = _fetch_whitespace(
            "https://users.nber.org/~rdehejia/data/nswre74_control.txt",
            cols,
        )
    else:
        cols = _LALONDE_COLS
        treated = _fetch_whitespace(
            "https://users.nber.org/~rdehejia/data/nsw_treated.txt",
            cols,
        )
        control_nsw = _fetch_whitespace(
            "https://users.nber.org/~rdehejia/data/nsw_control.txt",
            cols,
        )

    if external_control == "psid":
        ext = _fetch_whitespace(
            "https://users.nber.org/~rdehejia/data/psid_controls.txt",
            cols,
        )
    elif external_control == "cps":
        ext = _fetch_whitespace(
            "https://users.nber.org/~rdehejia/data/cps_controls.txt",
            cols,
        )
    else:
        raise ValueError(f"Unknown external_control: {external_control!r}")

    # Combine: keep NSW treated + external controls (the classic LaLonde setup)
    df = pd.concat([treated, ext], ignore_index=True)
    df = df.reset_index(drop=True)
    df.insert(0, "patient_id", [f"P{i}" for i in range(len(df))])
    df = df.rename(columns={"treatment": "exposure_status", "RE78": "outcome"})

    numeric = [c for c in ["age", "education", "RE74", "RE75"] if c in df.columns]
    categorical = [c for c in ["black", "hispanic", "married", "nodegree"] if c in df.columns]

    meta = {
        "features_numeric": numeric,
        "features_categorical": categorical,
    }
    df.attrs["meta"] = meta
    return df


# ---------------------------------------------------------------------------
# VLBW (Very Low Birth Weight) – O'Shea et al. 1992
# ---------------------------------------------------------------------------

_VLBW_URL = "https://hbiostat.org/data/repo/vlbw.zip"

_VLBW_NUMERIC = ["bwt", "gest", "apg1", "year"]

_VLBW_CATEGORICAL = ["race", "sex", "inout", "twn", "delivery"]


def load_vlbw() -> pd.DataFrame:
    """Download and return the VLBW dataset.

    Treatment: pneumo (pneumothorax, binary)
    Outcome: dead (binary mortality)
    ~622 complete cases after dropping rows with missing covariates.

    Returns DataFrame with patient_id, exposure_status, outcome, and covariates.
    """
    import zipfile, urllib.request

    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, "vlbw.csv")

    if not os.path.exists(cache_path):
        zip_path = os.path.join(_CACHE_DIR, "vlbw.zip")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(_VLBW_URL, zip_path)
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(_CACHE_DIR)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to download {_VLBW_URL}: {e}")

    df = pd.read_csv(cache_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Keep only rows with non-missing treatment
    df = df[df["pneumo"].notna()].copy()
    df["pneumo"] = df["pneumo"].astype(int)

    # Rename columns
    df = df.rename(columns={"pneumo": "exposure_status", "dead": "outcome"})

    # Drop rows with missing covariates
    all_covs = _VLBW_NUMERIC + _VLBW_CATEGORICAL
    keep = ["exposure_status", "outcome"] + all_covs
    df = df[keep].dropna().reset_index(drop=True)

    # Add patient_id
    df.insert(0, "patient_id", [f"P{i}" for i in range(len(df))])

    # Ensure numeric columns are numeric
    for col in _VLBW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode categoricals as strings for Gower
    for col in _VLBW_CATEGORICAL:
        df[col] = df[col].astype(str)

    meta = {
        "features_numeric": _VLBW_NUMERIC,
        "features_categorical": _VLBW_CATEGORICAL,
    }
    df.attrs["meta"] = meta
    return df


# ---------------------------------------------------------------------------
# IHDP (Infant Health and Development Program) – Hill 2011 / Shalit et al. 2017
# ---------------------------------------------------------------------------

_IHDP_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
_IHDP_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"

# Covariate descriptions (25 features, indices 0-24)
_IHDP_CONTINUOUS = ["birthweight", "birthhead", "birthlength", "gestage", "apgar1", "apgar5"]
_IHDP_BINARY = [
    "sex", "twin", "breech", "delivery", "fraclprem", "cigdrg", "alcohol",
    "momage", "momed", "married", "dmed", "prenatal", "workdur", "nnhealth",
    "infant_age", "sexofchild", "momwhite", "site1", "site2",
]


def _download_ihdp() -> tuple[str, str]:
    """Download IHDP NPZ files to cache, return paths."""
    import urllib.request
    os.makedirs(_CACHE_DIR, exist_ok=True)
    paths = {}
    for label, url in [("train", _IHDP_TRAIN_URL), ("test", _IHDP_TEST_URL)]:
        fname = os.path.join(_CACHE_DIR, os.path.basename(url))
        if not os.path.exists(fname):
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(url, fname)
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(f"Failed to download {url}: {e}")
        paths[label] = fname
    return paths["train"], paths["test"]


def load_ihdp_single(rep_idx: int = 0) -> pd.DataFrame:
    """Load a single IHDP replication as a DataFrame (train+test combined).

    Parameters
    ----------
    rep_idx : int
        Replication index (0-99).

    Returns DataFrame with:
      - patient_id: str
      - exposure_status: int (1=treated, 0=control)
      - outcome: float (cognitive test score, factual)
      - mu0: float (noiseless potential outcome under control)
      - mu1: float (noiseless potential outcome under treated)
      - 25 covariates (6 continuous + 19 binary)
    """
    import numpy as np
    train_path, test_path = _download_ihdp()

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    frames = []
    for split_idx, data in enumerate([train_data, test_data]):
        x = data["x"][:, :, rep_idx].copy()
        x[:, 13] -= 1  # Fix momage: {1,2} -> {0,1}

        t = data["t"][:, rep_idx].astype(int)
        yf = data["yf"][:, rep_idx]
        mu0 = data["mu0"][:, rep_idx]
        mu1 = data["mu1"][:, rep_idx]

        n = x.shape[0]
        df_split = pd.DataFrame(x, columns=_IHDP_CONTINUOUS + _IHDP_BINARY)
        df_split.insert(0, "patient_id", [f"IHDP-{split_idx}-{i}" for i in range(n)])
        df_split["exposure_status"] = t
        df_split["outcome"] = yf
        df_split["mu0"] = mu0
        df_split["mu1"] = mu1
        frames.append(df_split)

    df = pd.concat(frames, ignore_index=True)

    for col in _IHDP_BINARY:
        df[col] = df[col].astype(int).astype(str)

    meta = {
        "features_numeric": _IHDP_CONTINUOUS,
        "features_categorical": _IHDP_BINARY,
    }
    df.attrs["meta"] = meta
    return df


def load_ihdp(n_replications: int = 10):
    """Load multiple IHDP replications (train+test combined).

    Returns list of DataFrames, each with 1344 subjects, mu0/mu1 for true ATT.
    """
    return [load_ihdp_single(i) for i in range(n_replications)]


