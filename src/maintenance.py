"""Health states and maintenance policies.

Costs are in relative units: planned maintenance = 1, unplanned failure = 10.
Policies are compared by cost per 1000 flight cycles, so servicing too early
(fewer cycles flown) and failing (expensive) are both penalized.
"""
import numpy as np

C_MAINT, C_FAIL = 1.0, 10.0
LEAD = 10   # service when the interval's lower bound drops below this many cycles


def health_state(pred, lower, lead=LEAD):
    if lower < lead:
        return "Critical"
    if pred < 100:
        return "Degrading"
    return "Healthy"


def outcome(life, service_cycle, c_maint=C_MAINT, c_fail=C_FAIL):
    """(cost, cycles flown) for one engine that fails after `life` cycles."""
    if service_cycle is None or service_cycle >= life:
        return c_fail, life
    return c_maint, service_cycle


def cost_rate(outcomes):
    cost = sum(c for c, _ in outcomes)
    cycles = sum(n for _, n in outcomes)
    return 1000 * cost / cycles


def reactive(lives, **costs):
    """Run every engine until it fails."""
    return [outcome(life, None, **costs) for life in lives]


def fixed_interval(lives, T, **costs):
    """Service every engine after T cycles, whatever its condition."""
    return [outcome(life, T, **costs) for life in lives]


def best_interval(lives, **costs):
    """The fixed interval T with the lowest cost on the given (training) engines."""
    return min(range(20, max(lives)), key=lambda T: cost_rate(fixed_interval(lives, T, **costs)))


def predictive(lives, lowers, lead=LEAD, **costs):
    """Service each engine at the first cycle where its lower bound drops below `lead`.

    lowers[i] is the lower bound at every cycle of engine i (cycle 1, 2, ...).
    """
    results = []
    for life, lower in zip(lives, lowers):
        below = np.flatnonzero(lower < lead)
        service = int(below[0]) + 1 if len(below) else None
        results.append(outcome(life, service, **costs))
    return results
