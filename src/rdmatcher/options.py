"""Matching and execution settings."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal
import math


def _choice(name, value, choices):
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")


def _nonnegative(name, value, positive=False):
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'non-negative'}")


def _integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MatchingOptions:
    """Eligibility, distance weights and assignment priorities."""

    method: Literal['multi', 'propensity'] = 'multi'
    prioritize_coverage: bool = False
    gower_weights: Mapping[str, Any] | None = None
    propensity_col: str = 'propensity_logit'
    propensity_caliper: float | None = None
    ps_hybrid: bool = False
    ps_caliper: float = 0.2
    ps_caliper_strict: bool = True
    gower_sd_weights: bool = False
    gower_sd_reference: Literal['controls', 'pooled'] = 'controls'
    gower_sd_weights_mult: float = 1.96
    weight_numeric: float = 1.0
    weight_propensity: float = 1.0
    return_unmatched_cases: bool = True
    pca_filter: bool = False

    def __post_init__(self):
        _choice('method', self.method, {'multi', 'propensity'})
        _choice('gower_sd_reference', self.gower_sd_reference, {'controls', 'pooled'})
        for name in ('ps_caliper', 'weight_numeric', 'weight_propensity'):
            _nonnegative(name, getattr(self, name))
        _nonnegative('gower_sd_weights_mult', self.gower_sd_weights_mult, positive=True)
        if self.propensity_caliper is not None:
            _nonnegative('propensity_caliper', self.propensity_caliper)
        if not isinstance(self.propensity_col, str) or not self.propensity_col:
            raise ValueError('propensity_col must be a nonempty column name')
        if self.gower_weights is not None and not isinstance(self.gower_weights, Mapping):
            raise TypeError('gower_weights must map feature names to weights')
        if self.gower_sd_weights and self.gower_weights is not None:
            raise ValueError('Use explicit Gower weights or SD-calibrated weights, not both')


@dataclass(frozen=True)
class ExecutionOptions:
    """Resource limits and assignment solver selection."""

    n_jobs: int = 1
    batch_size: int = 10
    solver: Literal['scipy_sparse', 'hungarian', 'mcf'] = 'scipy_sparse'
    streaming: Literal['auto', 'on', 'off'] = 'auto'
    stream_block_size: int | None = None
    stream_threshold_gb: float = 1.0
    memory_limit_gb: float | None = None
    parallel_chunk_size: int | None = None

    def __post_init__(self):
        if self.n_jobs != -1:
            _integer('n_jobs', self.n_jobs)
        _integer('batch_size', self.batch_size)
        _choice('solver', self.solver, {'scipy_sparse', 'hungarian', 'mcf'})
        _choice('streaming', self.streaming, {'auto', 'on', 'off'})
        _nonnegative('stream_threshold_gb', self.stream_threshold_gb, positive=True)
        for name in ('stream_block_size', 'parallel_chunk_size'):
            value = getattr(self, name)
            if value is not None:
                _integer(name, value)
        if self.memory_limit_gb is not None:
            _nonnegative('memory_limit_gb', self.memory_limit_gb, positive=True)


@dataclass(frozen=True)
class MatchRequest:
    """Complete configuration for one graph-construction or matching call."""

    threshold: float
    ratio: int = 1
    distance_metric: str | None = None
    allocation: Literal['optimal', 'hybrid'] = 'optimal'
    matching: MatchingOptions = field(default_factory=MatchingOptions)
    execution: ExecutionOptions = field(default_factory=ExecutionOptions)

    def __post_init__(self):
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, (int, float)):
            raise TypeError('threshold must be a non-negative number')
        if math.isnan(self.threshold) or self.threshold < 0:
            raise ValueError('threshold must be non-negative')
        _integer('ratio', self.ratio)
        _choice('allocation', self.allocation, {'optimal', 'hybrid'})
        if self.distance_metric is not None and not isinstance(self.distance_metric, str):
            raise TypeError('distance_metric must be a string or None')
        if not isinstance(self.matching, MatchingOptions):
            raise TypeError('matching must be a MatchingOptions instance')
        if not isinstance(self.execution, ExecutionOptions):
            raise TypeError('execution must be an ExecutionOptions instance')
