"""CPU numerical sanity checks of three mathematical statements, not WM evidence."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm


def orthogonal_span(matrix, tol=1e-11):
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, s > tol * max(1.0, s[0])]


def observable_space(operators, cost, horizon):
    columns = [cost]
    frontier = [cost]
    for _ in range(horizon):
        frontier = [a.T @ v for a in operators for v in frontier]
        columns.extend(frontier)
    return orthogonal_span(np.stack(columns, axis=1))


def check_observables(seed=7):
    rng = np.random.default_rng(seed)
    d, horizon = 9, 3
    # Each transpose expands the goal space by at most one coordinate per step.
    operators = []
    for _ in range(2):
        a = np.diag(rng.uniform(.3, .7, d))
        a += np.diag(rng.uniform(.05, .3, d-1), 1)
        operators.append(a)
    c = np.eye(d)[0]
    v = observable_space(operators, c, horizon)
    small = [v.T @ a @ v for a in operators]
    error = 0.0
    for length in range(horizon + 1):
        for word in itertools.product(range(2), repeat=length):
            x = rng.normal(size=d)
            y = v.T @ x
            for j in word:
                x, y = operators[j] @ x, small[j] @ y
            error = max(error, abs(c @ x - (v.T @ c) @ y))
    return {"dimension": d, "horizon": horizon, "observable_rank": v.shape[1],
            "max_cost_error": float(error), "pass": bool(error < 1e-11)}


def check_locality():
    rows = []
    kappa, t, delta = .8, 1.3, .2
    for n in (8, 20, 60):
        m = np.diag(np.full(n-1, kappa), 1)
        for radius in (1, 3, 6):
            # x'=Mx+delta*e_R, x(0)=0; exact integral via augmented exponential.
            augmented = np.zeros((n+1, n+1))
            augmented[:n, :n] = m
            augmented[radius, n] = delta
            actual = abs(expm(t * augmented)[0, n])
            bound = delta * t * math.exp(kappa*t) * (kappa*t)**radius / math.factorial(radius+1)
            rows.append({"n": n, "radius": radius, "goal_error": float(actual),
                         "bound": bound, "pass": bool(actual <= bound + 1e-14)})
    return rows


def heisenberg_step(state, action, duration):
    x, y, q = state
    u, v = action
    return np.array([x+duration*u, y+duration*v, q+.5*duration*(x*v-y*u)])


def check_order():
    rows = []
    for h in (.01, .03, .1, .3):
        state = np.zeros(3)
        for action in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            state = heisenberg_step(state, action, h)
        rows.append({"h": h, "loop_q": float(state[2]), "expected": h*h,
                     "pass": bool(np.allclose(state, [0, 0, h*h], atol=1e-14))})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = {"scope": "synthetic mathematical checks; no trained checkpoint or simulator",
              "observable": check_observables(), "locality": check_locality(),
              "order": check_order()}
    result["pass"] = result["observable"]["pass"] and all(
        row["pass"] for key in ("locality", "order") for row in result[key])
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
