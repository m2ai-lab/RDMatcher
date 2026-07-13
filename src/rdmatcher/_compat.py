"""Compatibility helpers for third-party plotting libraries."""

from contextlib import contextmanager


def ensure_seaborn_pandas_compat() -> None:
    """Patch pandas option_context for seaborn/pandas 3 compatibility.

    seaborn<=0.13.2 still requests the removed pandas option
    ``mode.use_inf_as_na`` inside plotting internals. On pandas 3 this raises
    OptionError and breaks otherwise-valid plots. We keep seaborn plots and make
    that specific option a harmless no-op.
    """
    try:
        import pandas as pd
        from pandas._config.config import OptionError  # type: ignore
    except Exception:
        return

    original_option_context = pd.option_context

    # Avoid double-patching.
    if getattr(original_option_context, "__name__", "") == "_rdmatcher_safe_option_context":
        return

    @contextmanager
    def _noop_context():
        yield

    def _rdmatcher_safe_option_context(*args):
        try:
            return original_option_context(*args)
        except OptionError:
            if any(arg == "mode.use_inf_as_na" for arg in args[::2]):
                return _noop_context()
            raise

    _rdmatcher_safe_option_context.__name__ = "_rdmatcher_safe_option_context"
    pd.option_context = _rdmatcher_safe_option_context
