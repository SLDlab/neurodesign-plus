from __future__ import annotations

"""Legacy-style random order and ITI generators used by optimisation routines.

These helpers are retained for flat one-event design generation. Version 2
template-based designs build schedules in :mod:`neurodesign.classes`, but the
genetic search still relies on these utilities when constructing flat orders or
sampling classic ITI distributions.
"""

import numpy as np
import scipy
import scipy.stats as stats

from neurodesign import msequence


def order(
    nstim: int,
    ntrials: int,
    probabilities: list[float],
    ordertype: str,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """Sample a flat stimulus order for legacy one-event design modes.

    Parameters define the number of stimulus categories, desired sequence
    length, target category probabilities, and the sampling strategy
    (`random`, `blocked`, or `msequence`).
    """
    if ordertype not in ["random", "blocked", "msequence"]:
        raise ValueError(f"{ordertype} not known.")

    local_rng = rng if rng is not None else np.random.default_rng(seed)

    if ordertype == "blocked":
        blocksize = float(local_rng.choice(np.arange(1, 10)))
        nblocks = int(np.ceil(ntrials / blocksize))
        blockorder = _generate_order_items(probabilities, nblocks, local_rng)
        return np.repeat(blockorder, blocksize)[:ntrials].tolist()

    if ordertype == "msequence":
        seq = msequence.Msequence()
        seq.GenMseq(
            mLen=ntrials,
            stimtypeno=nstim,
            seed=seed if seed is not None else int(local_rng.integers(1, 2**31)),
        )
        idx = int(local_rng.integers(len(seq.orders)))
        return list(seq.orders[idx])

    return _generate_order_items(probabilities, ntrials, local_rng)


def _generate_order_items(probabilities, items, rng):
    """Draw categorical item identities according to ``probabilities``."""
    mult = rng.multinomial(1, probabilities, items)
    return [x.tolist().index(1) for x in mult]


def iti(
    ntrials: int,
    model: str,
    min: float | None = None,
    mean: float | None = None,
    max: float | None = None,
    lam=None,
    resolution: float = 0.1,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
):
    """Sample a legacy event-aligned ITI vector.

    The returned vector always has length ``ntrials`` and begins with ``0`` so
    that element ``i`` can be interpreted as the gap immediately preceding the
    ``i``-th event in the historical API.
    """
    local_rng = rng if rng is not None else np.random.default_rng(seed)

    if model == "fixed":
        smp = [0] + [mean] * (ntrials - 1)
        smp = resolution * np.round(np.array(smp) / resolution)
        return smp, lam

    if model == "uniform":
        mean = (min + max) / 2.0
        smp = local_rng.uniform(min, max, (ntrials - 1))
        smp = _fix_iti(smp, mean, min, max, resolution, local_rng)
        return np.append([0], smp), lam

    if model == "exponential":
        if not lam:
            lam = _compute_lambda(min, max, mean)
        smp = _rtexp((ntrials - 1), lam, min, max, local_rng)
        smp = _fix_iti(smp, mean, min, max, resolution, local_rng)
        return np.append([0], smp), lam

        raise ValueError(f"Unknown inter-trial interval model {model!r}")


def _fix_iti(smp, mean, min, max, resolution, rng):
    """Round sampled ITIs while nudging them back toward the requested mean."""
    smp = resolution * np.round(np.array(smp) / resolution)
    totaldiff = np.sum(smp) - mean * len(smp)
    while not np.isclose(totaldiff, 0, resolution) and np.mean(smp) > mean:
        chid = int(rng.integers(len(smp)))
        if (smp[chid] - min) < resolution or (max - smp[chid]) < resolution:
            continue
        smp[chid] = smp[chid] - np.sign(totaldiff) * resolution
        totaldiff = np.sum(smp) - mean * len(smp)
    return smp


def _compute_lambda(lower, upper, mean):
    """Infer the truncated-exponential scale parameter matching a bounded mean."""
    a = float(lower)
    b = float(upper)
    m = float(mean)
    opt = scipy.optimize.minimize(
        _difexp, 50, args=(a, b, m), bounds=((10 ** (-9), 100),), method="L-BFGS-B"
    )
    check_rng = np.random.default_rng(1000)
    check = _rtexp(100000, opt.x[0], lower, upper, check_rng)
    if not np.isclose(np.mean(check), mean, rtol=0.1):
        raise ValueError(
            "Error when figuring out lambda for exponential distribution: can't compute lambda."
        )
    return opt.x[0]


def _difexp(lam, lower, upper, mean):
    """Objective used when fitting a bounded exponential ITI distribution."""
    lam = float(np.asarray(lam).flat[0])
    diff = stats.truncexpon(
        (float(upper) - float(lower)) / lam, loc=float(lower), scale=lam
    ).mean() - float(mean)
    return abs(diff)


def _rtexp(ntrials, lam, lower, upper, rng):
    """Sample from a truncated exponential distribution on ``[lower, upper]``."""
    a = float(lower)
    b = float(upper)
    return stats.truncexpon((b - a) / lam, loc=a, scale=lam).rvs(
        ntrials, random_state=rng
    )
