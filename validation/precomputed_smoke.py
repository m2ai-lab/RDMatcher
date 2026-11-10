"""Self-contained smoke benchmark for dense precomputed RDM matching.

This is a package sanity check, not a manuscript performance benchmark. It
generates a seeded dense matrix, runs one RDM match, and prints phase timings
and certificate status as JSON. It never writes repository-local output.
"""
from __future__ import annotations

import argparse
import json
import resource
from time import perf_counter

import numpy as np
import pandas as pd

from rdmatcher import ExecutionOptions, RDMatcher


def rss_mib() -> float:
    """Return the Linux process high-water mark in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--controls", type=int, default=10_000)
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--threshold", type=float, default=float("inf"))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.cases < 1 or args.controls < 1 or args.ratio < 1:
        parser.error("--cases, --controls, and --ratio must be positive")
    if args.controls < args.cases * args.ratio:
        parser.error("--controls must be at least cases × ratio for complete matching")

    before = rss_mib()
    costs = np.random.default_rng(args.seed).uniform(0, 1, (args.cases, args.controls))
    population = pd.DataFrame({
        "id": np.arange(args.cases + args.controls),
        "treated": [1] * args.cases + [0] * args.controls,
    })
    input_rss = rss_mib()

    started = perf_counter()
    model = RDMatcher.from_distance_matrix(
        population, "id", "treated", costs,
        case_ids=np.arange(args.cases),
        control_ids=np.arange(args.cases, args.cases + args.controls),
    )
    initialization_sec = perf_counter() - started
    initialized_rss = rss_mib()

    started = perf_counter()
    model.match(
        args.threshold,
        ratio=args.ratio,
        execution=ExecutionOptions(n_jobs=args.n_jobs),
    )
    matching_sec = perf_counter() - started
    diagnostics = model.assignment_diagnostics_
    if not diagnostics.get("optimality_verified"):
        raise RuntimeError("RDM did not produce an exact optimality certificate")

    print(json.dumps({
        "cases": args.cases,
        "controls": args.controls,
        "ratio": args.ratio,
        "seed": args.seed,
        "threshold": "inf" if np.isposinf(args.threshold) else args.threshold,
        "n_jobs": args.n_jobs,
        "input_matrix_mib": costs.nbytes / 2**20,
        "initialization_sec": initialization_sec,
        "matching_sec": matching_sec,
        "candidate_preparation_sec": diagnostics["candidate_preparation_sec"],
        "allocation_sec": diagnostics["competitive_allocation_sec"]
        + diagnostics["global_assignment_sec"],
        "candidate_edges": diagnostics["candidate_edge_count"],
        "matched_controls": diagnostics["matched_control_count"],
        "requested_slots": args.cases * args.ratio,
        "optimality_verified": diagnostics["optimality_verified"],
        "process_peak_rss_mib": {
            "before_input": before,
            "after_input": input_rss,
            "after_initialization": initialized_rss,
            "after_matching": rss_mib(),
        },
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
