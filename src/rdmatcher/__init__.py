"""
RDMatcher
=========
RDMatcher is a Python package for matching methods with EHR data for epidemiological studies.
"""

__version__ = "0.1.01"

from ._compat import ensure_seaborn_pandas_compat

ensure_seaborn_pandas_compat()

from .RDMatcher import RDMatcher
