"""Independent exact-rational primal/residual check for precomputed Optimal."""
from fractions import Fraction
import hashlib
import math

_CERTIFICATE_MEMORY_BUDGET_BYTES = 128 * 2**20


def verify_assignment(precomputed, demands, coverage_priority, all_control_indices,
                      matches, threshold):
    """Certify a feasible assignment by exact residual-cost potentials.

    This reconstructs the objective from original supplied edge values, without
    using solver weights, binary scaling, or solver potentials. A feasible fixed
    flow is optimal iff its residual graph has no negative-cost cycle. A set of
    potentials making every residual reduced cost nonnegative certifies that
    condition. Penalties encode the declared lexicographic priorities exactly.
    """
    n_rows = len(precomputed)
    demands = [int(value) for value in demands]
    priorities = [bool(value) for value in coverage_priority]
    if len(demands) != n_rows or len(priorities) != n_rows:
        raise ValueError('Certificate demands and priorities must match candidate rows')
    edge_count = sum(len(row['candidate_positions']) for row in precomputed)
    if n_rows*256 + edge_count*512 > _CERTIFICATE_MEMORY_BUDGET_BYTES:
        raise ValueError('Exact optimality certificate exceeds its memory budget; no assignment was returned')
    costs = []
    for candidates in precomputed:
        if len(candidates['candidate_positions']) != len(candidates['candidate_distances']):
            raise ValueError('Certificate candidate positions and costs have different lengths')
        row = {}
        for position, value in zip(candidates['candidate_positions'], candidates['candidate_distances']):
            position, value = int(position), float(value)
            if position < 0 or position >= len(all_control_indices):
                raise ValueError('Certificate received an invalid control position')
            if not math.isfinite(value) or value < 0 or value > threshold:
                raise ValueError('Certificate received an ineligible edge')
            control = int(all_control_indices[position])
            if control in row:
                raise ValueError('Certificate received duplicate candidate coordinates')
            row[control] = Fraction.from_float(value)
        costs.append(row)

    if any(int(row) != row or row < 0 or row >= n_rows for row in matches):
        return dict(verified=False, reason='invalid_case_index')
    chosen, used, objective = [], set(), Fraction(0)
    for row in range(n_rows):
        raw = list(matches.get(row, []))
        if any(int(control) != control for control in raw):
            return dict(verified=False, reason='invalid_control_index')
        selected = [int(control) for control in raw]
        if len(selected) > demands[row] or len(set(selected)) != len(selected):
            return dict(verified=False, reason='invalid_case_demand')
        if any(control not in costs[row] or control in used for control in selected):
            return dict(verified=False, reason='ineligible_or_reused_control')
        chosen.append(set(selected))
        used.update(selected)
        objective += sum((costs[row][control] for control in selected), Fraction(0))

    controls = sorted({control for row in costs for control in row})
    column = {control: n_rows + i for i, control in enumerate(controls)}
    sink = n_rows + len(controls)
    nodes = sink + 1
    slots = sum(demands)
    span = slots * max((value for row in costs for value in row.values()), default=Fraction(0))
    extra_penalty = span + 1
    coverage_penalty = slots * extra_penalty + span + 1
    residual = []
    for row, candidates in enumerate(costs):
        for control, cost in candidates.items():
            residual.append((column[control], row, -cost) if control in chosen[row]
                            else (row, column[control], cost))
    for control in controls:
        residual.append((sink, column[control], Fraction(0)) if control in used
                        else (column[control], sink, Fraction(0)))

    def add_dummy(row, capacity, flow, cost):
        if flow < 0 or flow > capacity:
            raise ValueError('Certificate received an invalid dummy flow')
        if flow < capacity:
            residual.append((row, sink, cost))
        if flow:
            residual.append((sink, row, -cost))

    for row, demand in enumerate(demands):
        high_capacity = int(priorities[row] and demand > 0)
        high_flow = int(high_capacity and not chosen[row])
        add_dummy(row, high_capacity, high_flow, coverage_penalty)
        add_dummy(row, demand - high_capacity,
                  demand - len(chosen[row]) - high_flow, extra_penalty)

    # Bellman-Ford from an implicit zero-cost super-source reaches every node.
    potentials = [Fraction(0)] * nodes
    for iteration in range(nodes):
        changed = False
        for tail, head, cost in residual:
            proposed = potentials[tail] + cost
            if proposed < potentials[head]:
                potentials[head] = proposed
                changed = True
        if not changed:
            if any(cost + potentials[tail] - potentials[head] < 0
                   for tail, head, cost in residual):
                raise RuntimeError('Exact residual certificate has inconsistent potentials')
            digest = hashlib.sha256('\n'.join(str(value) for value in potentials).encode()).hexdigest()
            return dict(verified=True, method='exact_rational_residual_potentials',
                        nodes=nodes, residual_edges=len(residual), passes=iteration + 1,
                        potentials_sha256=digest, objective_numerator=str(objective.numerator),
                        objective_denominator=str(objective.denominator))
    return dict(verified=False, reason='negative_residual_cycle')
