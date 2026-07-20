from __future__ import annotations

"""Core schedule, metric, and optimisation classes for Neurodesign-Plus.

The version-2 architecture separates conceptual trials from flattened modeled
events. Conceptual trials define how schedules are sampled and where
between-trial intervals or rests may occur; flattened modeled events define the
regressor axis used for event timing, frequency metrics, and transition metrics.
"""

import copy
import functools
import hashlib
import json
import math
import shutil
import warnings
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.stats as stats
import sklearn.cluster
from numpy import transpose as t
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.special import gamma

from . import generate, report

REMOVED_TIMING_ARGUMENTS = {
    "t_pre": "trial_start_interval",
    "t_post": "post_event_interval",
    "stimuli_durations": "event_durations",
    "conditional_ITI": "event_transition_interval and/or inter_trial_interval",
    "ITImodel": "inter_trial_interval",
    "ITImin": "inter_trial_interval",
    "ITImean": "inter_trial_interval",
    "ITImax": "inter_trial_interval",
}


def progress_bar(text: str, color: str = "green") -> Progress:
    """Create a consistent Rich progress bar used by optimisation loops."""
    return Progress(
        TextColumn(f"[{color}]{text}"),
        SpinnerColumn("dots"),
        TimeElapsedColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )


def _json_key(value: Any) -> str:
    """Convert nested rule keys into stable JSON-serializable strings."""
    if isinstance(value, tuple):
        return json.dumps([_json_key(v) for v in value])
    return str(value)


def _display_rule(value: Any) -> Any:
    """Recursively normalize internal rule objects for human-readable export."""
    if isinstance(value, dict):
        return {str(k): _display_rule(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_display_rule(v) for v in value]
    if isinstance(value, tuple):
        return [_display_rule(v) for v in value]
    return value


def _json_default(value: Any) -> Any:
    """Convert numpy/scalar objects into stable JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _stable_json_bytes(value: Any) -> bytes:
    """Serialize nested design/specification payloads with deterministic ordering."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _find_new_resolution(TR, res):
    """Snap a requested resolution to a divisor of the repetition time."""
    n = TR * 1000.0
    divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(int(n / i))
    sorted_divisors = np.sort(divisors)
    resdivisor = TR / float(res)
    difs = np.abs(resdivisor - sorted_divisors)
    minind = np.where(difs == np.min(difs))[0]
    divisor = sorted_divisors[minind][0]
    return TR / divisor


def _round_to_resolution(inmat, res):
    """Round values down to the discrete modeling grid."""
    out = res * np.floor(np.array(inmat) / res)
    ind = out / res
    return out, [int(x) for x in ind]


def _round_scalar(value: float, res: float) -> float:
    """Round a scalar to the nearest modeling-grid step."""
    return float(res * np.round(float(value) / res))


def _ensure_non_negative(arg_name: str, value: float) -> float:
    """Validate that a timing parameter is finite and non-negative."""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{arg_name} must be finite and non-negative; got {value!r}")
    return float(value)


def _rule_id(prefix: str, selector_value: Any | None = None) -> str:
    """Build a readable provenance label for a resolved timing rule."""
    if selector_value is None:
        return prefix
    if isinstance(selector_value, tuple):
        selector_value = "->".join(str(v) for v in selector_value)
    return f"{prefix}[{selector_value}]"


@dataclass(frozen=True)
class NormalizedRule:
    """Canonical internal representation of one resolved timing distribution."""

    model: str
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the normalized rule."""
        out = {"model": self.model}
        if self.mean is not None:
            out["value" if self.model == "fixed" else "mean"] = self.mean
        if self.min is not None:
            out["min"] = self.min
        if self.max is not None:
            out["max"] = self.max
        if self.std is not None:
            out["std"] = self.std
        return out


@dataclass(frozen=True)
class SelectorSpec:
    """Canonical representation of selector-dispatched timing rules."""

    selector_kind: str
    rules: dict[Any, NormalizedRule]
    default: NormalizedRule | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the selector wrapper."""
        out = {}
        body = {str(k): v.as_dict() for k, v in self.rules.items()}
        if self.default is not None:
            body["default"] = self.default.as_dict()
        out[self.selector_kind] = body
        return out


def _validate_fixed_rule(arg_name: str, value: Any) -> NormalizedRule:
    """Normalize a scalar timing specification into a fixed rule."""
    return NormalizedRule(model="fixed", mean=_ensure_non_negative(arg_name, value))


@functools.cache
def _compute_truncated_exponential_scale(
    arg_name: str, mean: float, lower: float, upper: float
) -> float:
    """Infer the scale of a bounded exponential distribution with a target mean."""
    if mean < lower or mean > upper:
        raise ValueError(
            f"{arg_name} exponential rule has mean={mean} outside bounds [{lower}, {upper}]"
        )

    def objective(scale):
        scale = float(np.asarray(scale).flat[0])
        if scale <= 0:
            return 1e9
        dist = stats.truncexpon((upper - lower) / scale, loc=lower, scale=scale)
        return abs(dist.mean() - mean)

    result = scipy.optimize.minimize(
        objective,
        x0=np.array([max(mean - lower, 1e-6)]),
        bounds=((1e-9, 1e6),),
        method="L-BFGS-B",
    )
    scale = float(result.x[0])
    dist = stats.truncexpon((upper - lower) / scale, loc=lower, scale=scale)
    if not np.isclose(dist.mean(), mean, atol=1e-6, rtol=1e-6):
        raise ValueError(
            f"{arg_name} exponential rule has impossible bounded mean semantics"
        )
    return scale


@functools.cache
def _compute_truncated_normal_loc(
    arg_name: str, mean: float, std: float, lower: float, upper: float
) -> float:
    """Infer the latent Gaussian location for a bounded normal rule."""
    if mean < lower or mean > upper:
        raise ValueError(
            f"{arg_name} gaussian rule has mean={mean} outside bounds [{lower}, {upper}]"
        )

    def objective(loc):
        loc = float(np.asarray(loc).flat[0])
        a = (lower - loc) / std
        b = (upper - loc) / std
        return stats.truncnorm.mean(a, b, loc=loc, scale=std) - mean

    result = scipy.optimize.root_scalar(
        objective,
        bracket=(lower - 10 * std, upper + 10 * std),
        method="brentq",
    )
    if not result.converged:
        raise ValueError(
            f"{arg_name} gaussian rule has impossible bounded mean semantics"
        )
    return float(result.root)


def normalize_rule(spec: Any, arg_name: str) -> NormalizedRule | SelectorSpec:
    """Normalize user timing syntax into internal rule objects.

    The public API accepts scalars, explicit ``{'model': ...}`` dictionaries,
    and selector wrappers such as ``by_event_category`` or
    ``by_event_transition``. This function converts those variants into a
    canonical representation used by schedule generation.
    """
    if isinstance(spec, (int, float, np.integer, np.floating)):
        return _validate_fixed_rule(arg_name, float(spec))

    if not isinstance(spec, dict):
        raise TypeError(
            f"{arg_name} must be a scalar or dictionary rule; got {type(spec)!r}"
        )

    selector_keys = [k for k in spec.keys() if k.startswith("by_")]
    if "model" in spec and selector_keys:
        raise ValueError(f"{arg_name} mixes a global rule with selector wrapper syntax")
    if len(selector_keys) > 1:
        raise ValueError(f"{arg_name} may contain only one selector wrapper")

    if selector_keys:
        selector_kind = selector_keys[0]
        body = spec[selector_kind]
        if not isinstance(body, dict):
            raise TypeError(f"{arg_name}.{selector_kind} must be a dictionary")
        rules: dict[Any, NormalizedRule] = {}
        default = None
        for key, value in body.items():
            if key == "default":
                default = normalize_rule(value, f"{arg_name}.{selector_kind}.default")
                if isinstance(default, SelectorSpec):
                    raise ValueError(
                        f"{arg_name}.{selector_kind}.default may not nest selectors"
                    )
                continue
            normalized = normalize_rule(value, f"{arg_name}.{selector_kind}[{key!r}]")
            if isinstance(normalized, SelectorSpec):
                raise ValueError(
                    f"{arg_name}.{selector_kind}[{key!r}] may not nest selectors"
                )
            if selector_kind == "by_event_transition":
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(
                        f"{arg_name}.{selector_kind} keys must be ordered pairs; got {key!r}"
                    )
            rules[key] = normalized
        return SelectorSpec(selector_kind=selector_kind, rules=rules, default=default)

    model = spec.get("model")
    if model is None:
        raise ValueError(
            f"{arg_name} dictionary is ambiguous; use a scalar, a {{'model': ...}} rule, or a selector wrapper"
        )
    if model == "fixed":
        if "value" not in spec:
            raise ValueError(f"{arg_name} fixed rule requires 'value'")
        return _validate_fixed_rule(arg_name, spec["value"])
    if model == "uniform":
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"{arg_name} uniform rule requires 'min' and 'max'")
        lower = _ensure_non_negative(f"{arg_name}.min", spec["min"])
        upper = _ensure_non_negative(f"{arg_name}.max", spec["max"])
        if lower > upper:
            raise ValueError(f"{arg_name} uniform rule requires min <= max")
        if "mean" in spec:
            expected = (lower + upper) / 2.0
            if not np.isclose(expected, float(spec["mean"]), atol=1e-8, rtol=1e-8):
                raise ValueError(
                    f"{arg_name} uniform rule mean must equal (min + max) / 2; expected {expected}"
                )
        return NormalizedRule(
            model="uniform", mean=(lower + upper) / 2.0, min=lower, max=upper
        )
    if model == "exponential":
        if "mean" not in spec:
            raise ValueError(f"{arg_name} exponential rule requires 'mean'")
        mean = _ensure_non_negative(f"{arg_name}.mean", spec["mean"])
        lower = spec.get("min")
        upper = spec.get("max")
        lower_f = (
            _ensure_non_negative(f"{arg_name}.min", lower) if lower is not None else None
        )
        upper_f = (
            _ensure_non_negative(f"{arg_name}.max", upper) if upper is not None else None
        )
        if lower_f is not None and upper_f is not None and lower_f > upper_f:
            raise ValueError(f"{arg_name} exponential rule requires min <= max")
        if lower_f is not None and mean < lower_f:
            raise ValueError(
                f"{arg_name} exponential rule has mean={mean} outside bounds [{lower_f}, {upper_f}]"
            )
        if upper_f is not None and mean > upper_f:
            raise ValueError(
                f"{arg_name} exponential rule has mean={mean} outside bounds [{lower_f}, {upper_f}]"
            )
        return NormalizedRule(model="exponential", mean=mean, min=lower_f, max=upper_f)
    if model == "gaussian":
        if "mean" not in spec or "std" not in spec:
            raise ValueError(f"{arg_name} gaussian rule requires 'mean' and 'std'")
        mean = _ensure_non_negative(f"{arg_name}.mean", spec["mean"])
        std = float(spec["std"])
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"{arg_name}.std must be finite and > 0")
        lower = spec.get("min")
        upper = spec.get("max")
        if lower is None:
            raise ValueError(
                f"{arg_name} gaussian rules must define a non-negative 'min' bound"
            )
        lower_f = _ensure_non_negative(f"{arg_name}.min", lower)
        upper_f = (
            _ensure_non_negative(f"{arg_name}.max", upper) if upper is not None else None
        )
        if upper_f is not None and lower_f > upper_f:
            raise ValueError(f"{arg_name} gaussian rule requires min <= max")
        if mean < lower_f or (upper_f is not None and mean > upper_f):
            raise ValueError(
                f"{arg_name} gaussian rule has mean={mean} outside bounds [{lower_f}, {upper_f}]"
            )
        return NormalizedRule(
            model="gaussian", mean=mean, min=lower_f, max=upper_f, std=std
        )
    raise ValueError(f"{arg_name} uses unknown model {model!r}")


def sample_normalized_rule(
    rule: NormalizedRule, arg_name: str, rng: np.random.Generator, resolution: float
) -> float:
    """Sample one realized timing value from a normalized rule."""
    if rule.model == "fixed":
        value = float(rule.mean)
    elif rule.model == "uniform":
        value = float(rng.uniform(rule.min, rule.max))
    elif rule.model == "exponential":
        if rule.min is None and rule.max is None:
            value = float(rng.exponential(rule.mean))
        else:
            lower = 0.0 if rule.min is None else float(rule.min)
            upper = lower + 1e6 if rule.max is None else float(rule.max)
            scale = _compute_truncated_exponential_scale(
                arg_name, float(rule.mean), lower, upper
            )
            dist = stats.truncexpon((upper - lower) / scale, loc=lower, scale=scale)
            value = float(dist.rvs(random_state=rng))
    elif rule.model == "gaussian":
        lower = float(rule.min)
        upper = lower + 10 * float(rule.std) if rule.max is None else float(rule.max)
        loc = _compute_truncated_normal_loc(
            arg_name, float(rule.mean), float(rule.std), lower, upper
        )
        a = (lower - loc) / float(rule.std)
        b = (upper - loc) / float(rule.std)
        value = float(
            stats.truncnorm.rvs(a, b, loc=loc, scale=float(rule.std), random_state=rng)
        )
    else:
        raise ValueError(f"Unknown normalized rule model {rule.model!r}")
    return _round_scalar(_ensure_non_negative(arg_name, value), resolution)


class Design:
    """Represent one realized fMRI design with explicit event-level timing.

    The design stores the flattened modeled-event order, the realized schedule
    arrays derived from an :class:`Experiment`, and the computed efficiency
    metrics used for reporting and optimisation.
    """

    def __init__(
        self,
        experiment: Experiment | None = None,  # type: ignore[name-defined]
        schedule: dict[str, Any] | None = None,
        trial_sequence: list[Any] | None = None,
        template_sequence: list[str] | None = None,
    ):
        """Create a realized design from a fully materialized schedule.

        Parameters
        ----------
        experiment:
            Parent experiment specification that defines the modeling and
            optimisation context.
        schedule:
            Realized event-level schedule generated by the experiment.
        trial_sequence, template_sequence:
            Provenance for conceptual-trial sampling, used by resampling,
            crossover, mutation, and validation exports.
        """
        self.experiment = experiment
        self.Fe = 0.0
        self.Fd = 0.0
        self.Ff = 0.0
        self.Fc = 0.0
        self.F = 0.0
        self.schedule = None
        self.order = None
        self.trial_sequence = trial_sequence
        self.template_sequence = template_sequence

        if self.experiment is None:
            raise ValueError("Design requires an Experiment")

        if schedule is not None:
            self._apply_schedule(schedule)
        else:
            raise ValueError(
                "Design construction requires a realized schedule. "
                "Use Experiment.create_design() or Experiment.create_manual_design(...) in version 2.0."
            )

    def _apply_schedule(self, schedule: dict[str, Any]):
        """Hydrate convenience arrays from the realized schedule dictionary."""
        self.schedule = copy.deepcopy(schedule)
        self.order = list(schedule["order"])
        self.n_events = len(self.order)
        self.event_categories = list(schedule["event_categories"])
        self.realized_event_durations = np.array(
            schedule["realized_event_durations"], dtype=float
        )
        self.trial_ids = np.array(schedule["trial_ids"], dtype=int)
        self.event_index_within_trial = np.array(
            schedule["event_index_within_trial"], dtype=int
        )
        self.trial_template_ids = list(schedule["trial_template_ids"])
        self.trial_type_ids = list(schedule["trial_type_ids"])
        self.realized_trial_start_intervals = np.array(
            schedule["realized_trial_start_intervals"], dtype=float
        )
        self.realized_post_event_intervals = np.array(
            schedule["realized_post_event_intervals"], dtype=float
        )
        self.realized_event_transition_intervals = np.array(
            schedule["realized_event_transition_intervals"], dtype=float
        )
        self.realized_inter_trial_intervals = np.array(
            schedule["realized_inter_trial_intervals"], dtype=float
        )
        self.realized_rest_intervals = np.array(
            schedule["realized_rest_intervals"], dtype=float
        )
        self.event_onsets = np.array(schedule["event_onsets"], dtype=float)
        self.event_offsets = np.array(schedule["event_offsets"], dtype=float)
        self.trial_starts = np.array(schedule["trial_starts"], dtype=float)
        self.trial_ends = np.array(schedule["trial_ends"], dtype=float)
        self.trial_start_event_index = np.array(
            schedule["trial_start_event_index"], dtype=int
        )
        self.trial_end_event_index = np.array(
            schedule["trial_end_event_index"], dtype=int
        )
        self.within_trial_transition_from_event_index = np.array(
            schedule["within_trial_transition_from_event_index"], dtype=int
        )
        self.within_trial_transition_to_event_index = np.array(
            schedule["within_trial_transition_to_event_index"], dtype=int
        )
        self.inter_trial_boundary_after_trial = np.array(
            schedule["inter_trial_boundary_after_trial"], dtype=int
        )
        self.selector_provenance = copy.deepcopy(schedule["selector_provenance"])
        self.schedule_table = copy.deepcopy(schedule["schedule_table"])

    def check_maxrep(self, maxrep):
        """Return ``False`` if any flattened event category repeats too often."""
        # Tracks consecutive run length directly on the category codes,
        # rather than concatenating codes into a string and searching for a
        # repeated substring: with 10+ categories, codes become multi-digit,
        # so e.g. category 1 followed by category 12 concatenates to "112",
        # which a substring search would misread as three consecutive 1's.
        if maxrep is None:
            return True
        run_value = None
        run_length = 0
        for code in self.order:
            if code == run_value:
                run_length += 1
            else:
                run_value = code
                run_length = 1
            if run_length >= maxrep:
                return False
        return True

    def check_hardprob(self):
        """Check whether the realized flattened event proportions match ``P``."""
        if (
            len(self.experiment.P) != self.experiment.n_stimuli
            or len(self.experiment.P) == 0
        ):
            return False
        counts = np.zeros(self.experiment.n_stimuli, dtype=float)
        for code in self.order:
            if not isinstance(code, (int, np.integer)):
                return False
            code = int(code)
            if code < 0 or code >= self.experiment.n_stimuli:
                return False
            counts[code] += 1.0
        total = counts.sum()
        if total <= 0:
            return False
        obsprob = counts / total
        close = np.isclose(
            np.array(self.experiment.P, dtype=float), obsprob, atol=0.001, rtol=0.0
        )
        return bool(np.all(close))

    def spawn_resampled_timing(self, rng: np.random.Generator) -> Design:
        """Resample timing while preserving the same conceptual-trial structure."""
        schedule = self.experiment.realize_from_trial_sequence(
            trial_sequence=self.trial_sequence,
            rng=rng,
            template_sequence=self.template_sequence,
        )
        return Design(experiment=self.experiment, schedule=schedule)

    def crossover(self, other, seed=1234):
        """Create offspring designs by recombining order or template sequences."""
        rng = np.random.default_rng(seed)
        if self.experiment.mode in {"flat_fixed_order", "fixed_trials"}:
            return [self.spawn_resampled_timing(rng), other.spawn_resampled_timing(rng)]

        if self.experiment.mode == "template_sampled":
            assert self.trial_sequence is not None and other.trial_sequence is not None
            changepoint = int(rng.integers(0, len(self.trial_sequence)))
            seq1 = list(self.trial_sequence[:changepoint]) + list(
                other.trial_sequence[changepoint:]
            )
            seq2 = list(other.trial_sequence[:changepoint]) + list(
                self.trial_sequence[changepoint:]
            )
            child1 = self.experiment.realize_from_trial_sequence(seq1, rng, seq1)
            child2 = self.experiment.realize_from_trial_sequence(seq2, rng, seq2)
            return [
                Design(
                    experiment=self.experiment,
                    schedule=child1,
                    trial_sequence=seq1,
                    template_sequence=seq1,
                ),
                Design(
                    experiment=self.experiment,
                    schedule=child2,
                    trial_sequence=seq2,
                    template_sequence=seq2,
                ),
            ]

        changepoint = int(rng.integers(0, len(self.order)))
        offspringorder1 = list(self.order)[:changepoint] + list(other.order)[changepoint:]
        offspringorder2 = list(other.order)[:changepoint] + list(self.order)[changepoint:]
        child1 = self.experiment.realize_flat_order(offspringorder1, rng)
        child2 = self.experiment.realize_flat_order(offspringorder2, rng)
        return [
            Design(
                experiment=self.experiment,
                schedule=child1,
                trial_sequence=offspringorder1,
            ),
            Design(
                experiment=self.experiment,
                schedule=child2,
                trial_sequence=offspringorder2,
            ),
        ]

    def mutation(self, q, seed=1234):
        """Randomly perturb a design while respecting the active design mode."""
        rng = np.random.default_rng(seed)
        if self.experiment.mode in {"flat_fixed_order", "fixed_trials"}:
            return self.spawn_resampled_timing(rng)

        if self.experiment.mode == "template_sampled":
            seq = list(self.trial_sequence)
            nmut = max(1, int(len(seq) * q)) if len(seq) > 0 else 0
            if nmut > 0:
                idxs = rng.choice(len(seq), size=nmut, replace=False)
                for idx in np.atleast_1d(idxs):
                    seq[int(idx)] = self.experiment.sample_template_id(rng)
            schedule = self.experiment.realize_from_trial_sequence(seq, rng, seq)
            return Design(
                experiment=self.experiment,
                schedule=schedule,
                trial_sequence=seq,
                template_sequence=seq,
            )

        mutated = list(self.order)
        nmut = max(1, int(len(mutated) * q)) if len(mutated) > 0 else 0
        if nmut > 0:
            idxs = rng.choice(len(mutated), size=nmut, replace=False)
            for idx in np.atleast_1d(idxs):
                mutated[int(idx)] = int(rng.integers(self.experiment.n_stimuli))
        schedule = self.experiment.realize_flat_order(mutated, rng)
        return Design(
            experiment=self.experiment, schedule=schedule, trial_sequence=mutated
        )

    def designmatrix(self):
        """Build event-level and convolved design matrices for this schedule.

        ``Xnonconv`` represents the realized modeled-event occupancy on the scan
        grid. ``Xconv`` is the HRF-convolved event-regressor matrix used by the
        estimation efficiency metrics.
        """
        onsetX, XindStim = _round_to_resolution(
            self.event_onsets, self.experiment.resolution
        )
        event_durs, _ = _round_to_resolution(
            self.realized_event_durations, self.experiment.resolution
        )
        stim_duration_tp = [
            int(round(float(x) / self.experiment.resolution)) for x in event_durs
        ]
        max_endpoint = max(
            XindStim[i] + stim_duration_tp[i] for i in range(len(self.order))
        )
        if max_endpoint > self.experiment.n_tp:
            return False

        X_X = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])
        for i, stim in enumerate(self.order):
            onset = XindStim[i]
            dur = stim_duration_tp[i]
            for j in range(dur):
                t_idx = onset + j
                if 0 <= t_idx < self.experiment.n_tp:
                    X_X[t_idx, int(stim)] = 1

        deconvM = np.zeros(
            [
                self.experiment.n_tp,
                int(self.experiment.laghrf * self.experiment.n_stimuli),
            ]
        )
        for stim in range(self.experiment.n_stimuli):
            for j in range(min(int(self.experiment.laghrf), self.experiment.n_tp)):
                deconvM[j:, self.experiment.laghrf * stim + j] = X_X[
                    : (self.experiment.n_tp - j), stim
                ]

        idxX = [
            int(x)
            for x in np.arange(
                0, self.experiment.n_tp, self.experiment.TR / self.experiment.resolution
            )
        ]
        if len(idxX) - self.experiment.white.shape[0] == 1:
            idxX = idxX[: self.experiment.white.shape[0]]

        deconvMdown = deconvM[idxX, :]
        Xwhite = np.dot(np.dot(t(deconvMdown), self.experiment.white), deconvMdown)

        X_Z = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])
        for stim in range(self.experiment.n_stimuli):
            X_Z[:, stim] = deconvM[
                :, (stim * self.experiment.laghrf) : ((stim + 1) * self.experiment.laghrf)
            ].dot(self.experiment.basishrf)
        X_Z = X_Z[idxX, :]
        X_X = X_X[idxX, :]
        Zwhite = t(X_Z) @ self.experiment.white @ X_Z

        self.X = Xwhite
        self.Z = Zwhite
        self.Xconv = X_Z
        self.Xnonconv = X_X
        self.CX = self.experiment.CX
        self.C = self.experiment.C
        return self

    def FeCalc(self, Aoptimality=True):
        """Compute estimation-efficiency score ``Fe`` for the deconvolved model."""
        try:
            invM = scipy.linalg.inv(self.X)
        except scipy.linalg.LinAlgError:
            try:
                invM = scipy.linalg.pinv(self.X)
            except scipy.linalg.LinAlgError:
                # self.X (Xwhite) is symmetric by construction (A^T W A); when
                # pinv's general SVD fails to converge on a near-singular
                # matrix (platform/LAPACK-dependent), the symmetric
                # eigendecomposition-based pseudo-inverse is a numerically
                # more stable equivalent for this case.
                invM = scipy.linalg.pinvh(self.X)
        invM = np.array(invM)
        CMC = np.dot(np.dot(self.CX, invM), t(self.CX))
        if Aoptimality:
            self.Fe = float(self.CX.shape[0] / np.trace(CMC))
        else:
            self.Fe = float(np.linalg.det(CMC) ** (-1 / len(self.C)))
        self.Fe = self.Fe / self.experiment.FeMax
        return self

    def FdCalc(self, Aoptimality=True):
        """Compute detection-efficiency score ``Fd`` for the convolved model."""
        try:
            invM = scipy.linalg.inv(self.Z)
        except scipy.linalg.LinAlgError:
            try:
                invM = scipy.linalg.pinv(self.Z)
            except scipy.linalg.LinAlgError:
                # See FeCalc: self.Z (Zwhite) is symmetric by construction,
                # so fall back to the eigendecomposition-based pseudo-inverse
                # when general-SVD pinv fails to converge.
                invM = scipy.linalg.pinvh(self.Z)
        invM = np.array(invM)
        CMC = self.C @ invM @ t(self.C)
        if Aoptimality:
            self.Fd = float(len(self.C) / np.trace(CMC))
        else:
            self.Fd = float(np.linalg.det(CMC) ** (-1 / len(self.C)))
        self.Fd = self.Fd / self.experiment.FdMax
        return self

    def _frequency_mismatch(self) -> float:
        """Measure deviation between observed and expected flattened event counts."""
        event_count = len(self.order)
        trialcount = Counter(self.order)
        observed_counts = np.array(
            [trialcount.get(x, 0) for x in range(self.experiment.n_stimuli)],
            dtype=float,
        )
        expected_counts = float(event_count) * np.array(self.experiment.P, dtype=float)
        return float(np.sum(np.abs(observed_counts - expected_counts)))

    def _transition_mismatch(self, confoundorder=3) -> float:
        """Measure deviation from expected flattened transition counts."""
        event_count = len(self.order)
        Q = np.zeros(
            [self.experiment.n_stimuli, self.experiment.n_stimuli, confoundorder]
        )
        for n in range(event_count):
            for r in np.arange(1, confoundorder + 1):
                if n > (r - 1):
                    Q[self.order[n], self.order[n - r], r - 1] += 1
        Qexp = np.zeros(
            [self.experiment.n_stimuli, self.experiment.n_stimuli, confoundorder]
        )
        for si in range(self.experiment.n_stimuli):
            for sj in range(self.experiment.n_stimuli):
                for r in np.arange(1, confoundorder + 1):
                    Qexp[si, sj, r - 1] = (
                        self.experiment.P[si] * self.experiment.P[sj] * (event_count + 1)
                    )
        return float(np.sum(np.abs(Q - Qexp)))

    def FcCalc(self, confoundorder=3):
        """Compute transition-balance score ``Fc`` on the flattened event axis."""
        event_count = len(self.order)
        Qmatch = self._transition_mismatch(confoundorder)
        fc_max = self.experiment.fc_max_for_event_count(event_count, confoundorder)
        self.Fc = 1.0 if np.isclose(fc_max, 0.0) else 1 - (Qmatch / fc_max)
        return self

    def FfCalc(self):
        """Compute frequency-balance score ``Ff`` on the flattened event axis."""
        event_count = len(self.order)
        mismatch = self._frequency_mismatch()
        ff_max = self.experiment.ff_max_for_event_count(event_count)
        self.Ff = 1.0 if np.isclose(ff_max, 0.0) else 1 - mismatch / ff_max
        return self

    def FCalc(self, weights, Aoptimality=True, confoundorder=3):
        """Compute all requested component scores and their weighted objective."""
        if weights[0] > 0:
            self.FeCalc(Aoptimality)
        if weights[1] > 0:
            self.FdCalc(Aoptimality)
        self.FfCalc()
        self.FcCalc(confoundorder)
        matr = np.array([self.Fe, self.Fd, self.Ff, self.Fc])
        self.F = float(np.sum(np.array(weights) * matr))
        return self

    def export_schedule(self) -> list[dict[str, Any]]:
        """Return the row-wise event schedule used for reports and validation."""
        return copy.deepcopy(self.schedule_table)

    def export_payload(self) -> dict[str, Any]:
        """Export schedule arrays, counts, and metrics for downstream artifacts."""
        return {
            "counts": {
                "n_conceptual_trials": int(self.experiment.n_conceptual_trials),
                "n_events": int(self.n_events),
            },
            "schedule": self.export_schedule(),
            "schedule_arrays": copy.deepcopy(self.schedule),
            "metrics": {
                "F": float(self.F),
                "Fe": float(self.Fe),
                "Fd": float(self.Fd),
                "Ff": float(self.Ff),
                "Fc": float(self.Fc),
            },
        }

    def stable_hash(self) -> str:
        """Return a deterministic hash of the exported realized design payload."""
        return hashlib.sha256(_stable_json_bytes(self.export_payload())).hexdigest()


class Experiment:
    """Store the experiment specification and generate realized designs.

    Version 2 distinguishes conceptual trials from flattened modeled events:
    conceptual-trial counts govern sampling and trial boundaries, while the
    realized event axis governs event-level timing and Ff/Fc calculations.
    """

    def __init__(
        self,
        TR: float,
        P,
        C,
        rho: float,
        n_stimuli: int,
        stim_duration=None,
        event_durations=None,
        trial_start_interval=0.0,
        post_event_interval=0.0,
        event_transition_interval=0.0,
        inter_trial_interval=0.0,
        rest_every_n_trials=None,
        rest_interval=0.0,
        n_trials: int | None = None,
        duration=None,
        resolution=0.1,
        FeMax=1,
        FdMax=1,
        FcMax=1,
        FfMax=1,
        maxrep=None,
        hardprob=False,
        confoundorder=3,
        order=None,
        trial_templates=None,
        trials=None,
        trial_template_probabilities=None,
        n_conceptual_trials=None,
        seed: int | None = None,
        ordertype: str = "random",
        restnum=None,
        restdur=None,
        trial_max=None,
        **kwargs,
    ):
        """Create an experiment specification.

        The constructor accepts either classic flat one-event designs (via
        ``order``/``n_trials``, or generated from ``P``) or template-based
        conceptual-trial specifications (via ``trial_templates`` plus either
        ``trials`` or ``trial_template_probabilities``/``n_conceptual_trials``).
        In template modes, ``n_conceptual_trials`` controls sampling and
        boundaries, while the realized flattened event sequence determines
        event-level timing and Ff/Fc metrics.

        Parameters
        ----------
        TR:
            Repetition time (scan interval), in seconds.
        P:
            Target proportion of each modeled event category (length
            ``n_stimuli``, sums to ~1). Used for flat-order generation and,
            when ``hardprob`` is set, as a hard constraint on realized
            proportions; otherwise it is the soft reference for ``Ff``.
        C:
            Contrast matrix (one row per contrast, one column per modeled
            event category) used by ``Fe``/``Fd`` efficiency calculations.
        rho:
            AR(1) autocorrelation coefficient used to build the temporal
            whitening matrix for ``Fd``.
        n_stimuli:
            Number of distinct modeled event categories.
        stim_duration:
            Legacy alias for a fixed ``event_durations`` value; ignored if
            ``event_durations`` is also given.
        event_durations:
            Duration rule for modeled events: a scalar, a rule dict (for
            example ``{"model": "fixed", "value": 1.0}``), or a selector
            keyed by event category. Ignored for template modes, where each
            template event carries its own ``duration``.
        trial_start_interval, post_event_interval, event_transition_interval,
        inter_trial_interval, rest_interval:
            Timing-rule specifications (scalar or rule dict) for, respectively:
            the interval before a trial's first event, the interval after each
            event, the interval between events inside the same conceptual
            trial, the interval after a trial ends, and the interval inserted
            at optional rest boundaries. See ``MIGRATION_2.0.md`` for the full
            legacy-name mapping (for example ``t_pre`` -> ``trial_start_interval``).
        rest_every_n_trials:
            Insert a rest interval after every ``N`` conceptual trials;
            ``None`` disables rests.
        n_trials:
            Number of flat one-event trials to generate when ``order`` is not
            given and no trial templates are configured.
        duration:
            Total experiment duration in seconds. If ``None``, it is derived
            from the scheduling mode's trial/interval structure.
        resolution:
            Time-grid step (seconds) used for the modeling grid; must divide
            ``TR`` evenly (adjusted with a warning if it does not).
        FeMax, FdMax, FcMax, FfMax:
            Calibration references used to scale ``Fe``/``Fd``/``Fc``/``Ff``
            into a comparable range. Left at their default of ``1`` (i.e.
            uncalibrated), the corresponding raw score is unbounded; the
            optimisation prerun calibrates ``FeMax``/``FdMax`` automatically
            when that metric has positive weight, and ``FfMax``/``FcMax`` are
            calibrated analytically at construction time.
        maxrep:
            Maximum allowed number of consecutive repeats of the same event
            category in the flattened order; ``None`` disables the check.
        hardprob:
            If ``True``, require the realized flattened event proportions to
            closely match ``P`` (hard constraint) instead of only scoring the
            mismatch softly via ``Ff``.
        confoundorder:
            Maximum lag order considered by the ``Fc`` transition-balance
            score.
        order:
            Explicit flat one-event category sequence. If given, designs are
            realized from this fixed order rather than sampled.
        trial_templates:
            List of template dicts for conceptual-trial modes, each with
            ``template_id``, ``trial_type``, and an ``events`` list of
            ``{"category", "code", "duration"}`` entries.
        trials:
            Explicit fixed sequence of template references (by
            ``template_id``), used instead of probabilistic sampling.
        trial_template_probabilities:
            Sampling probability for each entry in ``trial_templates``, used
            with ``n_conceptual_trials`` to draw a random trial sequence.
        n_conceptual_trials:
            Number of conceptual trials to sample when
            ``trial_template_probabilities`` drives trial-sequence sampling.
        seed:
            Base seed for all deterministic RNG streams derived from this
            experiment; defaults to ``1234`` if not given.
        ordertype:
            Flat-order generation strategy: ``"random"``, ``"blocked"``, or
            ``"msequence"``.
        trial_max:
            Optional informational expected-maximum trial duration (seconds),
            shown in generated reports; not enforced during scheduling.
        """
        for old_name, replacement in REMOVED_TIMING_ARGUMENTS.items():
            if old_name in kwargs:
                raise TypeError(
                    f"{old_name!r} was removed in neurodesign-plus 2.0; use {replacement!r} instead"
                )
        if restnum is not None or restdur is not None:
            raise TypeError(
                "restnum/restdur were removed in neurodesign-plus 2.0; use rest_every_n_trials/rest_interval"
            )

        self.TR = float(TR)
        self.P = np.array(P, dtype=float)
        self.C = np.array(C, dtype=float)
        self.rho = float(rho)
        self.n_stimuli = int(n_stimuli)
        self.resolution = float(resolution)
        self.maxrep = maxrep
        self.hardprob = hardprob
        self.confoundorder = confoundorder
        self.FeMax = FeMax
        self.FdMax = FdMax
        self.FcMax = FcMax
        self.FfMax = FfMax
        self.seed = seed if seed is not None else 1234
        self.ordertype = ordertype
        self.duration = duration
        self.n_trials = n_trials
        self.n_conceptual_trials = n_conceptual_trials
        self.trial_max = trial_max
        self.stim_duration = stim_duration
        self.requested_event_durations = copy.deepcopy(
            event_durations
            if event_durations is not None
            else stim_duration if stim_duration is not None else 1.0
        )

        if not np.isclose(self.TR % self.resolution, 0):
            self.resolution = _find_new_resolution(self.TR, self.resolution)
            warnings.warn(
                "the resolution is adjusted to be a multiple of the TR. "
                f"New resolution: {self.resolution}"
            )

        self.trial_start_interval_requested = copy.deepcopy(trial_start_interval)
        self.post_event_interval_requested = copy.deepcopy(post_event_interval)
        self.event_transition_interval_requested = copy.deepcopy(
            event_transition_interval
        )
        self.inter_trial_interval_requested = copy.deepcopy(inter_trial_interval)
        self.rest_interval_requested = copy.deepcopy(rest_interval)
        self.rest_every_n_trials = rest_every_n_trials

        self.order = order
        self.order_fixed = order is not None
        self.trial_templates_public = copy.deepcopy(trial_templates)
        self.trials_public = copy.deepcopy(trials)
        self.trial_template_probabilities = copy.deepcopy(trial_template_probabilities)

        self._resolve_mode()
        self._prepare_categories()
        self.event_duration_spec = self._normalize_event_duration_spec(
            self.requested_event_durations
        )
        self.trial_start_interval_spec = normalize_rule(
            trial_start_interval, "trial_start_interval"
        )
        self.post_event_interval_spec = normalize_rule(
            post_event_interval, "post_event_interval"
        )
        self.event_transition_interval_spec = normalize_rule(
            event_transition_interval, "event_transition_interval"
        )
        self.inter_trial_interval_spec = normalize_rule(
            inter_trial_interval, "inter_trial_interval"
        )
        self.rest_interval_spec = normalize_rule(rest_interval, "rest_interval")
        self._validate_mode_specific_semantics()
        self._prepare_templates_and_trials()
        self.countstim()
        self.CreateTsComp()
        self.CreateLmComp()
        self._ff_max_cache: dict[int, float] = {}
        self._fc_max_cache: dict[tuple[int, int], float] = {}
        self.max_eff()

    def _resolve_mode(self):
        """Infer which scheduling mode is active from the provided inputs."""
        has_templates = self.trial_templates_public is not None
        has_trials = self.trials_public is not None
        has_probs = self.trial_template_probabilities is not None
        has_n_conceptual = self.n_conceptual_trials is not None
        if self.order is not None and (
            has_templates or has_trials or has_probs or has_n_conceptual
        ):
            raise ValueError(
                "order is mutually exclusive with template-based trial inputs"
            )
        if has_trials and not has_templates:
            raise ValueError("trials requires trial_templates")
        if has_probs or has_n_conceptual:
            if not (has_templates and has_probs and has_n_conceptual):
                raise ValueError(
                    "probabilistic template mode requires trial_templates, trial_template_probabilities, and n_conceptual_trials"
                )
            if has_trials:
                raise ValueError(
                    "trials may not be combined with probabilistic template sampling"
                )
            self.mode = "template_sampled"
        elif has_trials:
            self.mode = "fixed_trials"
        elif self.order is not None:
            self.mode = "flat_fixed_order"
        else:
            self.mode = "flat_generated"
            if self.n_trials is None:
                raise ValueError("flat one-event generation requires n_trials")

    def _prepare_categories(self):
        """Resolve modeled event-category labels and integer codes."""
        if self.mode in {"fixed_trials", "template_sampled"}:
            explicit_codes = {}
            categories = []
            for template in self.trial_templates_public:
                for event in template["events"]:
                    category = event["category"]
                    categories.append(category)
                    if "code" in event:
                        explicit_codes[category] = int(event["code"])
            seen = []
            for category in categories:
                if category not in seen:
                    seen.append(category)
            if explicit_codes:
                if len(explicit_codes) != len(seen):
                    raise ValueError(
                        "every template event category must define an explicit code or none may"
                    )
                if sorted(explicit_codes.values()) != list(range(self.n_stimuli)):
                    raise ValueError(
                        "explicit template event codes must cover 0..n_stimuli-1"
                    )
                ordered = [None] * self.n_stimuli
                for category, code in explicit_codes.items():
                    ordered[code] = category
                self.category_labels = ordered
            else:
                self.category_labels = list(seen)
            if len(self.category_labels) != self.n_stimuli:
                raise ValueError(
                    "n_stimuli must match the number of unique modeled event categories"
                )
            self.category_to_index = {
                label: idx for idx, label in enumerate(self.category_labels)
            }
        else:
            self.category_labels = list(range(self.n_stimuli))
            self.category_to_index = {idx: idx for idx in range(self.n_stimuli)}

    def _normalize_event_duration_spec(self, spec):
        """Normalize event-duration rules for flat or template-based designs."""
        if (
            self.mode in {"fixed_trials", "template_sampled"}
            and self.trial_templates_public is not None
        ):
            per_category: dict[Any, Any] = {}
            for template in self.trial_templates_public:
                for event in template["events"]:
                    category = event["category"]
                    duration_spec = event.get("duration", spec)
                    if (
                        category in per_category
                        and per_category[category] != duration_spec
                    ):
                        raise ValueError(
                            f"event category {category!r} has conflicting duration specifications across templates"
                        )
                    per_category[category] = duration_spec
            return {
                "selector_kind": "by_event_category",
                "rules": {
                    category: normalize_rule(rule, f"event_durations[{category!r}]")
                    for category, rule in per_category.items()
                },
            }
        if isinstance(spec, list):
            if len(spec) != self.n_stimuli:
                raise ValueError("event_durations list length must match n_stimuli")
            rules = {}
            for idx, rule in enumerate(spec):
                rules[
                    (
                        idx
                        if self.mode in {"flat_generated", "flat_fixed_order"}
                        else self.category_labels[idx]
                    )
                ] = normalize_rule(rule, f"event_durations[{idx}]")
            return {"selector_kind": "by_event_category", "rules": rules}
        normalized = normalize_rule(spec, "event_durations")
        if isinstance(normalized, SelectorSpec):
            return {
                "selector_kind": normalized.selector_kind,
                "rules": normalized.rules,
                "default": normalized.default,
            }
        return normalized

    def _validate_mode_specific_semantics(self):
        """Reject unsupported timing combinations for the chosen mode."""
        if isinstance(self.inter_trial_interval_spec, SelectorSpec):
            raise ValueError(
                "inter_trial_interval does not support selector wrappers in version 2.0"
            )
        if isinstance(self.rest_interval_spec, SelectorSpec):
            raise ValueError(
                "rest_interval does not support selector wrappers in version 2.0"
            )
        if self.rest_every_n_trials is None and self.rest_interval_requested not in (
            0,
            0.0,
            {"model": "fixed", "value": 0.0},
        ):
            if not (
                isinstance(self.rest_interval_requested, (int, float))
                and float(self.rest_interval_requested) == 0.0
            ):
                raise ValueError("rest_interval requires rest_every_n_trials")
        if self.rest_every_n_trials is not None and (
            not isinstance(self.rest_every_n_trials, int) or self.rest_every_n_trials <= 0
        ):
            raise ValueError("rest_every_n_trials must be a positive integer")

    def _prepare_templates_and_trials(self):
        """Normalize conceptual-trial templates and fixed trial sequences."""
        self.templates_by_id: dict[str, dict[str, Any]] = {}
        self.template_ids: list[str] = []
        if self.trial_templates_public is not None:
            for template in self.trial_templates_public:
                template_id = template["template_id"]
                if template_id in self.templates_by_id:
                    raise ValueError(f"duplicate template_id {template_id!r}")
                if not template.get("events"):
                    raise ValueError(
                        f"template {template_id!r} must define at least one event"
                    )
                normalized_events = []
                for event_index, event in enumerate(template["events"]):
                    category = event["category"]
                    if category not in self.category_to_index:
                        raise ValueError(
                            f"unknown event category {category!r} in template {template_id!r}"
                        )
                    duration_spec = event.get("duration", self.requested_event_durations)
                    rule = normalize_rule(
                        duration_spec,
                        f"trial_templates[{template_id!r}].events[{event_index}].duration",
                    )
                    if isinstance(rule, SelectorSpec):
                        raise ValueError(
                            "event duration selectors inside template events are not supported"
                        )
                    normalized_events.append(
                        {
                            "category": category,
                            "code": self.category_to_index[category],
                            "duration_rule": rule,
                        }
                    )
                self.templates_by_id[template_id] = {
                    "template_id": template_id,
                    "trial_type": template.get("trial_type", template_id),
                    "events": normalized_events,
                }
                self.template_ids.append(template_id)
        if self.mode == "fixed_trials":
            self.fixed_trial_sequence = []
            for idx, trial in enumerate(self.trials_public):
                template_id = trial["template_id"]
                if template_id not in self.templates_by_id:
                    raise ValueError(
                        f"trial {idx} references unknown template_id {template_id!r}"
                    )
                self.fixed_trial_sequence.append(template_id)
            self.n_conceptual_trials = len(self.fixed_trial_sequence)
        elif self.mode == "template_sampled":
            if len(self.trial_template_probabilities) != len(self.template_ids):
                raise ValueError(
                    "trial_template_probabilities must align with trial_templates"
                )
            probs = np.array(self.trial_template_probabilities, dtype=float)
            if np.any(probs < 0) or not np.isclose(probs.sum(), 1.0):
                raise ValueError(
                    "trial_template_probabilities must be non-negative and sum to 1"
                )
            self.template_probabilities = probs
        else:
            self.fixed_trial_sequence = None
            if self.order is not None:
                self.n_conceptual_trials = len(self.order)

    def make_design_rng(self, salt: int = 0) -> np.random.Generator:
        """Create a reproducible RNG derived from the experiment seed."""
        ss = np.random.SeedSequence([self.seed, salt])
        return np.random.default_rng(ss)

    def sample_template_id(self, rng: np.random.Generator) -> str:
        """Sample one template identifier from the configured template weights."""
        idx = int(rng.choice(len(self.template_ids), p=self.template_probabilities))
        return self.template_ids[idx]

    def sample_trial_sequence(self, rng: np.random.Generator) -> list[str]:
        """Sample a full conceptual-trial template sequence."""
        return [self.sample_template_id(rng) for _ in range(self.n_conceptual_trials)]

    def _spec_max(self, spec: NormalizedRule | SelectorSpec | dict[str, Any]) -> float:
        """Return a conservative upper bound for a timing rule."""
        if isinstance(spec, NormalizedRule):
            if spec.model == "fixed":
                return float(spec.mean)
            if spec.model == "uniform":
                return float(spec.max)
            if spec.model == "exponential":
                return float(
                    spec.max if spec.max is not None else max(spec.mean * 4, spec.mean)
                )
            if spec.model == "gaussian":
                return float(
                    spec.max if spec.max is not None else spec.mean + 4 * spec.std
                )
        if isinstance(spec, SelectorSpec):
            values = [self._spec_max(rule) for rule in spec.rules.values()]
            if spec.default is not None:
                values.append(self._spec_max(spec.default))
            return max(values) if values else 0.0
        if isinstance(spec, dict) and "rules" in spec:
            return max(self._spec_max(rule) for rule in spec["rules"].values())
        raise TypeError(f"Unsupported spec for max extraction: {type(spec)!r}")

    def _resolve_rule(
        self, spec, selector_value, arg_name: str
    ) -> tuple[NormalizedRule, str]:
        """Resolve one selector-dispatched rule and its provenance label."""
        if isinstance(spec, NormalizedRule):
            return spec, _rule_id(arg_name)
        if isinstance(spec, SelectorSpec):
            if selector_value in spec.rules:
                return spec.rules[selector_value], _rule_id(arg_name, selector_value)
            if spec.default is not None:
                return spec.default, _rule_id(arg_name, "default")
            raise ValueError(
                f"{arg_name} has no rule for selector {selector_value!r} and no default"
            )
        if isinstance(spec, dict) and spec.get("selector_kind") == "by_event_category":
            rules = spec["rules"]
            if selector_value in rules:
                rule = rules[selector_value]
                if isinstance(rule, SelectorSpec):
                    raise ValueError(f"{arg_name} nested selector is not supported")
                return rule, _rule_id(arg_name, selector_value)
            default = spec.get("default")
            if default is not None:
                return default, _rule_id(arg_name, "default")
            raise ValueError(
                f"{arg_name} has no rule for event category {selector_value!r}"
            )
        raise TypeError(f"Unsupported rule specification for {arg_name}")

    def _sample_value(
        self, spec, selector_value, arg_name: str, rng: np.random.Generator
    ) -> tuple[float, str]:
        """Sample one realized timing value and keep its provenance identifier."""
        rule, rule_id = self._resolve_rule(spec, selector_value, arg_name)
        return sample_normalized_rule(rule, arg_name, rng, self.resolution), rule_id

    def _estimate_flat_trial_max(self) -> float:
        """Estimate the maximum event duration for flat one-event trial modes."""
        if isinstance(self.event_duration_spec, NormalizedRule):
            event_max = self._spec_max(self.event_duration_spec)
        else:
            event_max = max(
                self._spec_max(rule)
                for rule in self.event_duration_spec["rules"].values()
            )
        return event_max

    def countstim(self):
        """Compute duration summaries implied by the active scheduling mode."""
        if self.mode in {"flat_generated", "flat_fixed_order"}:
            self.n_conceptual_trials = (
                self.n_trials if self.n_trials is not None else len(self.order)
            )
            self.trial_duration = (
                self._estimate_flat_trial_max()
                + self._spec_max(self.trial_start_interval_spec)
                + self._spec_max(self.post_event_interval_spec)
            )
            inter_trial_max = self._spec_max(self.inter_trial_interval_spec)
            rest_max = self._spec_max(self.rest_interval_spec)
            total = self.n_conceptual_trials * self.trial_duration
            total += max(self.n_conceptual_trials - 1, 0) * inter_trial_max
            if self.rest_every_n_trials:
                total += (
                    (self.n_conceptual_trials - 1) // self.rest_every_n_trials
                ) * rest_max
            self.duration = total if self.duration is None else self.duration
            return

        trial_durations = []
        for template_id in self.template_ids:
            template = self.templates_by_id[template_id]
            total = self._spec_max(self.trial_start_interval_spec)
            nevents = len(template["events"])
            for event_idx, event in enumerate(template["events"]):
                total += self._spec_max(event["duration_rule"])
                total += self._spec_max(self.post_event_interval_spec)
                if event_idx < nevents - 1:
                    total += self._spec_max(self.event_transition_interval_spec)
            trial_durations.append(total)
        inter_trial_max = self._spec_max(self.inter_trial_interval_spec)
        rest_max = self._spec_max(self.rest_interval_spec)
        max_trial_duration = max(trial_durations) if trial_durations else 0.0
        self.trial_duration = max_trial_duration
        total = self.n_conceptual_trials * max_trial_duration
        total += max(self.n_conceptual_trials - 1, 0) * inter_trial_max
        if self.rest_every_n_trials:
            total += (
                (self.n_conceptual_trials - 1) // self.rest_every_n_trials
            ) * rest_max
        self.duration = total if self.duration is None else self.duration

    def CreateTsComp(self):
        """Build scan-grid and modeling-grid time bases."""
        self.n_scans = int(np.ceil(self.duration / self.TR))
        self.n_tp = int(np.ceil(self.duration / self.resolution))
        self.r_scans = np.arange(0, self.duration, self.TR)
        self.r_tp = np.arange(0, self.duration, self.resolution)
        return self

    def CreateLmComp(self):
        """Build HRF, drift, and whitening components used by design scoring."""
        self.canonical()
        self.CX = np.array(np.kron(self.C, np.eye(self.laghrf)))
        self.S = np.asarray(self.drift(np.arange(0, self.n_scans)))
        base = [1 + self.rho**2, -1 * self.rho] + [0] * (self.n_scans - 2)
        self.V2 = scipy.linalg.toeplitz(base)
        self.V2[0, 0] = 1
        self.V2[self.n_scans - 1, self.n_scans - 1] = 1
        self.V2 = np.asarray(self.V2)
        self.white = (
            self.V2
            - self.V2
            @ t(self.S)
            @ np.linalg.pinv(self.S @ self.V2 @ t(self.S))
            @ self.S
            @ self.V2
        )
        return self

    def canonical(self):
        """Construct the canonical SPM-style HRF basis on the modeling grid."""
        p = [6, 16, 1, 1, 6, 0, 32]
        dt = self.resolution
        s = np.array(range(int(np.ceil(p[6] / dt))))
        hrf = (
            self.spm_Gpdf(s, p[0] / p[2], dt / p[2])
            - self.spm_Gpdf(s, p[1] / p[3], dt / p[3]) / p[4]
        )
        self.basishrf = hrf / np.sum(hrf)
        self.durhrf = p[6]
        self.laghrf = int(np.ceil(self.durhrf / self.resolution))
        return self

    @staticmethod
    def drift(s, deg=3):
        """Return polynomial drift regressors evaluated at sample locations."""
        S = np.ones([deg, len(s)])
        s = np.array(s)
        tmpt = np.array(2.0 * s / float(len(s) - 1) - 1)
        S[1] = tmpt
        for k in np.arange(2, deg):
            S[k] = ((2.0 * k - 1.0) / k) * tmpt * S[k - 1] - ((k - 1) / float(k)) * S[
                k - 2
            ]
        return S

    @staticmethod
    def spm_Gpdf(s, h, l):
        """Evaluate the gamma density used by the canonical HRF."""
        s = np.array(s)
        out = np.zeros_like(s, dtype=float)
        positive = s > 0
        res = (
            (h - 1) * np.log(s[positive])
            + h * np.log(l)
            - l * s[positive]
            - np.log(gamma(h))
        )
        out[positive] = np.exp(res)
        return out

    def _null_order_for_event_count(self, event_count: int) -> list[int]:
        """Construct the worst-case single-category null order for calibration."""
        return [int(np.argmin(self.P))] * int(event_count)

    def ff_max_for_event_count(self, event_count: int) -> float:
        """Return the ``Ff`` normalization constant for a given event count."""
        event_count = int(event_count)
        cached = self._ff_max_cache.get(event_count)
        if cached is not None:
            return cached
        null_order = self._null_order_for_event_count(event_count)
        trialcount = Counter(null_order)
        observed_counts = np.array(
            [trialcount.get(x, 0) for x in range(self.n_stimuli)], dtype=float
        )
        expected_counts = float(event_count) * np.array(self.P, dtype=float)
        mismatch = float(np.sum(np.abs(observed_counts - expected_counts)))
        self._ff_max_cache[event_count] = mismatch
        return mismatch

    def fc_max_for_event_count(
        self, event_count: int, confoundorder: int | None = None
    ) -> float:
        """Return the ``Fc`` normalization constant for a given event count."""
        event_count = int(event_count)
        confoundorder = (
            self.confoundorder if confoundorder is None else int(confoundorder)
        )
        cache_key = (event_count, confoundorder)
        cached = self._fc_max_cache.get(cache_key)
        if cached is not None:
            return cached
        null_order = self._null_order_for_event_count(event_count)
        Q = np.zeros([self.n_stimuli, self.n_stimuli, confoundorder])
        for n in range(event_count):
            for r in np.arange(1, confoundorder + 1):
                if n > (r - 1):
                    Q[null_order[n], null_order[n - r], r - 1] += 1
        Qexp = np.zeros([self.n_stimuli, self.n_stimuli, confoundorder])
        for si in range(self.n_stimuli):
            for sj in range(self.n_stimuli):
                for r in np.arange(1, confoundorder + 1):
                    Qexp[si, sj, r - 1] = self.P[si] * self.P[sj] * (event_count + 1)
        mismatch = float(np.sum(np.abs(Q - Qexp)))
        self._fc_max_cache[cache_key] = mismatch
        return mismatch

    def max_eff(self):
        """Initialize cached normalization constants for the active experiment."""
        if self.mode == "template_sampled":
            return self
        if self.mode == "flat_generated":
            event_count = self.n_trials
        elif self.mode == "flat_fixed_order":
            event_count = len(self.order)
        elif self.mode == "fixed_trials":
            event_count = sum(
                len(self.templates_by_id[trial["template_id"]]["events"])
                for trial in self.trials_public
            )
        else:
            event_count = self.n_trials
        self.FfMax = self.ff_max_for_event_count(event_count)
        self.FcMax = self.fc_max_for_event_count(event_count, self.confoundorder)
        return self

    def realize_manual_flat_design(
        self, order, inter_trial_intervals, all_event_durations=None
    ):
        """Materialize a flat one-event schedule from explicit event inputs."""
        if self.mode not in {"flat_generated", "flat_fixed_order"}:
            raise ValueError(
                "manual design construction is supported only for flat one-event designs"
            )
        if len(order) != len(inter_trial_intervals):
            raise ValueError(
                "manual flat design requires one event-aligned inter-trial value per event"
            )
        rng = self.make_design_rng(500)
        if all_event_durations is None:
            event_durations = [
                self._sample_value(
                    self.event_duration_spec, code, "event_durations", rng
                )[0]
                for code in order
            ]
        else:
            event_durations = [
                _round_scalar(float(x), self.resolution) for x in all_event_durations
            ]
        trial_start_intervals = []
        post_event_intervals = []
        for code in order:
            trial_start_intervals.append(
                self._sample_value(
                    self.trial_start_interval_spec, code, "trial_start_interval", rng
                )[0]
            )
            post_event_intervals.append(
                self._sample_value(
                    self.post_event_interval_spec, code, "post_event_interval", rng
                )[0]
            )
        schedule = self._build_flat_schedule(
            order=list(order),
            event_durations=event_durations,
            trial_start_intervals=trial_start_intervals,
            post_event_intervals=post_event_intervals,
            inter_trial_intervals=[
                _round_scalar(float(x), self.resolution)
                for x in inter_trial_intervals[1:]
            ],
            rest_intervals=[0.0] * max(len(order) - 1, 0),
            selector_provenance={
                "event_duration_rule_ids": [
                    _rule_id("event_durations", code) for code in order
                ],
                "trial_start_rule_ids": [_rule_id("trial_start_interval") for _ in order],
                "post_event_rule_ids": [_rule_id("post_event_interval") for _ in order],
                "inter_trial_rule_ids": [
                    _rule_id("manual_inter_trial") for _ in range(max(len(order) - 1, 0))
                ],
                "rest_rule_ids": [
                    _rule_id("rest_interval") for _ in range(max(len(order) - 1, 0))
                ],
                "event_transition_rule_ids": [],
            },
        )
        schedule["legacy_event_aligned_inter_trial"] = list(inter_trial_intervals)
        return schedule

    def create_manual_design(
        self,
        order: Sequence[int],
        inter_trial_intervals: Sequence[float],
        event_durations: Sequence[float] | None = None,
    ) -> Design:
        """Wrap a manually specified flat schedule in a :class:`Design`."""
        schedule = self.realize_manual_flat_design(
            order=list(order),
            inter_trial_intervals=list(inter_trial_intervals),
            all_event_durations=(
                None if event_durations is None else list(event_durations)
            ),
        )
        return Design(experiment=self, schedule=schedule, trial_sequence=list(order))

    def create_design(self, seed: int | None = None) -> Design:
        """Sample or realize one design under the active scheduling mode."""
        rng = self.make_design_rng(0 if seed is None else seed)
        if self.mode == "flat_fixed_order":
            schedule = self.realize_flat_order(self.order, rng)
            return Design(
                experiment=self, schedule=schedule, trial_sequence=list(self.order)
            )
        if self.mode == "fixed_trials":
            schedule = self.realize_from_trial_sequence(
                self.fixed_trial_sequence, rng, self.fixed_trial_sequence
            )
            return Design(
                experiment=self,
                schedule=schedule,
                trial_sequence=list(self.fixed_trial_sequence),
                template_sequence=list(self.fixed_trial_sequence),
            )
        if self.mode == "template_sampled":
            template_sequence = self.sample_trial_sequence(rng)
            schedule = self.realize_from_trial_sequence(
                template_sequence, rng, template_sequence
            )
            return Design(
                experiment=self,
                schedule=schedule,
                trial_sequence=list(template_sequence),
                template_sequence=list(template_sequence),
            )
        order = generate.order(
            self.n_stimuli,
            self.n_trials,
            self.P.tolist(),
            ordertype=self.ordertype,
            rng=rng,
        )
        schedule = self.realize_flat_order(order, rng)
        return Design(experiment=self, schedule=schedule, trial_sequence=list(order))

    def export_specification(self) -> dict[str, Any]:
        """Export the experiment specification with separated trial/event counts."""
        if self.mode == "fixed_trials":
            n_events = sum(
                len(self.templates_by_id[trial["template_id"]]["events"])
                for trial in self.trials_public
            )
        elif self.mode in {"flat_generated", "flat_fixed_order"}:
            n_events = self.n_trials if self.mode == "flat_generated" else len(self.order)
        else:
            n_events = None
        return {
            "mode": self.mode,
            "TR": self.TR,
            "P": self.P.tolist(),
            "C": self.C.tolist(),
            "rho": self.rho,
            "n_stimuli": self.n_stimuli,
            "resolution": self.resolution,
            "n_trials": self.n_trials,
            "n_conceptual_trials": self.n_conceptual_trials,
            "n_events": n_events,
            "event_durations_requested": _display_rule(self.requested_event_durations),
            "trial_start_interval_requested": _display_rule(
                self.trial_start_interval_requested
            ),
            "post_event_interval_requested": _display_rule(
                self.post_event_interval_requested
            ),
            "event_transition_interval_requested": _display_rule(
                self.event_transition_interval_requested
            ),
            "inter_trial_interval_requested": _display_rule(
                self.inter_trial_interval_requested
            ),
            "rest_interval_requested": _display_rule(self.rest_interval_requested),
            "rest_every_n_trials": self.rest_every_n_trials,
            "order": copy.deepcopy(self.order),
            "trial_templates": _display_rule(self.trial_templates_public),
            "trials": _display_rule(self.trials_public),
            "trial_template_probabilities": copy.deepcopy(
                self.trial_template_probabilities
            ),
            "seed": self.seed,
        }

    def specification_hash(self) -> str:
        """Return a deterministic hash of the exported experiment specification."""
        return hashlib.sha256(_stable_json_bytes(self.export_specification())).hexdigest()

    def realize_flat_order(self, order, rng: np.random.Generator):
        """Realize timing arrays for a classic flat one-event order."""
        trial_start_intervals = []
        post_event_intervals = []
        event_durations = []
        event_duration_rule_ids = []
        trial_start_rule_ids = []
        post_event_rule_ids = []
        for code in order:
            event_value, event_rule_id = self._sample_value(
                self.event_duration_spec, code, "event_durations", rng
            )
            start_value, start_rule_id = self._sample_value(
                self.trial_start_interval_spec, code, "trial_start_interval", rng
            )
            post_value, post_rule_id = self._sample_value(
                self.post_event_interval_spec, code, "post_event_interval", rng
            )
            event_durations.append(event_value)
            trial_start_intervals.append(start_value)
            post_event_intervals.append(post_value)
            event_duration_rule_ids.append(event_rule_id)
            trial_start_rule_ids.append(start_rule_id)
            post_event_rule_ids.append(post_rule_id)
        inter_trial_intervals = []
        inter_trial_rule_ids = []
        rest_intervals = []
        rest_rule_ids = []
        for boundary in range(max(len(order) - 1, 0)):
            value, rule_id = self._sample_value(
                self.inter_trial_interval_spec, None, "inter_trial_interval", rng
            )
            inter_trial_intervals.append(value)
            inter_trial_rule_ids.append(rule_id)
            if (
                self.rest_every_n_trials
                and (boundary + 1) % self.rest_every_n_trials == 0
            ):
                rest_value, rest_rule_id = self._sample_value(
                    self.rest_interval_spec, None, "rest_interval", rng
                )
            else:
                rest_value, rest_rule_id = 0.0, _rule_id("rest_interval", "none")
            rest_intervals.append(rest_value)
            rest_rule_ids.append(rest_rule_id)
        schedule = self._build_flat_schedule(
            order=list(order),
            event_durations=event_durations,
            trial_start_intervals=trial_start_intervals,
            post_event_intervals=post_event_intervals,
            inter_trial_intervals=inter_trial_intervals,
            rest_intervals=rest_intervals,
            selector_provenance={
                "event_duration_rule_ids": event_duration_rule_ids,
                "trial_start_rule_ids": trial_start_rule_ids,
                "post_event_rule_ids": post_event_rule_ids,
                "inter_trial_rule_ids": inter_trial_rule_ids,
                "rest_rule_ids": rest_rule_ids,
                "event_transition_rule_ids": [],
            },
        )
        return schedule

    def _build_flat_schedule(
        self,
        order,
        event_durations,
        trial_start_intervals,
        post_event_intervals,
        inter_trial_intervals,
        rest_intervals,
        selector_provenance,
    ):
        """Build a schedule dictionary for flat one-event designs."""
        T = len(order)
        trial_ids = list(range(T))
        event_index_within_trial = [0] * T
        trial_template_ids = [None] * T
        trial_type_ids = [self.category_labels[code] for code in order]
        trial_starts = []
        trial_ends = []
        event_onsets = []
        event_offsets = []
        cursor = 0.0
        schedule_table = []
        for trial_idx, code in enumerate(order):
            trial_start = cursor
            onset = trial_start + trial_start_intervals[trial_idx]
            offset = onset + event_durations[trial_idx]
            trial_end = offset + post_event_intervals[trial_idx]
            trial_starts.append(trial_start)
            event_onsets.append(onset)
            event_offsets.append(offset)
            trial_ends.append(trial_end)
            following_transition = None
            following_inter_trial = (
                inter_trial_intervals[trial_idx]
                if trial_idx < len(inter_trial_intervals)
                else None
            )
            following_rest = (
                rest_intervals[trial_idx] if trial_idx < len(rest_intervals) else None
            )
            schedule_table.append(
                {
                    "run_event_index": trial_idx,
                    "trial_id": trial_idx,
                    "trial_index": trial_idx,
                    "trial_template_id": None,
                    "trial_type_id": self.category_labels[code],
                    "event_index_within_trial": 0,
                    "event_category": self.category_labels[code],
                    "event_code": int(code),
                    "trial_start": trial_start,
                    "realized_trial_start_interval": trial_start_intervals[trial_idx],
                    "event_onset": onset,
                    "realized_event_duration": event_durations[trial_idx],
                    "event_offset": offset,
                    "realized_post_event_interval": post_event_intervals[trial_idx],
                    "following_event_transition_interval": following_transition,
                    "following_inter_trial_interval": following_inter_trial,
                    "following_rest_interval": following_rest,
                    "trial_end": trial_end,
                    "event_duration_rule_id": selector_provenance[
                        "event_duration_rule_ids"
                    ][trial_idx],
                    "trial_start_rule_id": selector_provenance["trial_start_rule_ids"][
                        trial_idx
                    ],
                    "post_event_rule_id": selector_provenance["post_event_rule_ids"][
                        trial_idx
                    ],
                    "event_transition_rule_id": None,
                    "inter_trial_rule_id": (
                        selector_provenance["inter_trial_rule_ids"][trial_idx]
                        if trial_idx < len(inter_trial_intervals)
                        else None
                    ),
                    "rest_rule_id": (
                        selector_provenance["rest_rule_ids"][trial_idx]
                        if trial_idx < len(rest_intervals)
                        else None
                    ),
                }
            )
            cursor = trial_end
            if trial_idx < len(inter_trial_intervals):
                cursor += inter_trial_intervals[trial_idx] + rest_intervals[trial_idx]
        return {
            "order": order,
            "event_categories": [self.category_labels[code] for code in order],
            "realized_event_durations": event_durations,
            "trial_ids": trial_ids,
            "event_index_within_trial": event_index_within_trial,
            "trial_template_ids": trial_template_ids,
            "trial_type_ids": trial_type_ids,
            "trial_start_event_index": list(range(T)),
            "trial_end_event_index": list(range(T)),
            "realized_trial_start_intervals": trial_start_intervals,
            "trial_starts": trial_starts,
            "trial_ends": trial_ends,
            "event_onsets": event_onsets,
            "event_offsets": event_offsets,
            "realized_post_event_intervals": post_event_intervals,
            "within_trial_transition_from_event_index": [],
            "within_trial_transition_to_event_index": [],
            "realized_event_transition_intervals": [],
            "inter_trial_boundary_after_trial": list(range(max(T - 1, 0))),
            "realized_inter_trial_intervals": inter_trial_intervals,
            "realized_rest_intervals": rest_intervals,
            "legacy_event_aligned_inter_trial": [0.0] + list(inter_trial_intervals),
            "selector_provenance": selector_provenance,
            "schedule_table": schedule_table,
        }

    def realize_from_trial_sequence(
        self, trial_sequence, rng: np.random.Generator, template_sequence=None
    ):
        """Realize a full event-level schedule from conceptual-trial templates."""
        trial_sequence = list(trial_sequence)
        order = []
        event_categories = []
        event_durations = []
        trial_ids = []
        event_index_within_trial = []
        trial_template_ids = []
        trial_type_ids = []
        trial_start_event_index = []
        trial_end_event_index = []
        trial_start_intervals = []
        trial_start_rule_ids = []
        post_event_intervals = []
        post_event_rule_ids = []
        transition_from = []
        transition_to = []
        transition_intervals = []
        transition_rule_ids = []
        event_duration_rule_ids = []
        event_onsets = []
        event_offsets = []
        trial_starts = []
        trial_ends = []
        schedule_table = []
        inter_trial_intervals = []
        inter_trial_rule_ids = []
        rest_intervals = []
        rest_rule_ids = []
        cursor = 0.0
        global_event_index = 0
        transition_lookup: dict[tuple[int, int], float] = {}
        transition_rule_lookup: dict[tuple[int, int], str] = {}

        for trial_idx, template_id in enumerate(trial_sequence):
            template = self.templates_by_id[template_id]
            trial_template_ids.append(template_id)
            trial_type_ids.append(template["trial_type"])
            trial_start, start_rule_id = self._sample_value(
                self.trial_start_interval_spec,
                template["trial_type"],
                "trial_start_interval",
                rng,
            )
            trial_start_intervals.append(trial_start)
            trial_start_rule_ids.append(start_rule_id)
            trial_start_event_index.append(global_event_index)
            trial_starts.append(cursor)
            event_cursor = cursor + trial_start
            events = template["events"]
            for event_idx, event in enumerate(events):
                code = int(event["code"])
                category = event["category"]
                duration = sample_normalized_rule(
                    event["duration_rule"],
                    "event_durations",
                    rng,
                    self.resolution,
                )
                duration_rule_id = _rule_id("event_durations", category)
                post_value, post_rule_id = self._sample_value(
                    self.post_event_interval_spec, category, "post_event_interval", rng
                )

                order.append(code)
                event_categories.append(category)
                event_durations.append(duration)
                trial_ids.append(trial_idx)
                event_index_within_trial.append(event_idx)
                event_onsets.append(event_cursor)
                event_offset = event_cursor + duration
                event_offsets.append(event_offset)
                post_event_intervals.append(post_value)
                post_event_rule_ids.append(post_rule_id)
                event_duration_rule_ids.append(duration_rule_id)

                following_transition = None
                following_transition_rule_id = None
                if event_idx < len(events) - 1:
                    next_category = events[event_idx + 1]["category"]
                    transition_value, transition_rule_id = self._sample_value(
                        self.event_transition_interval_spec,
                        (category, next_category),
                        "event_transition_interval",
                        rng,
                    )
                    transition_from.append(global_event_index)
                    transition_to.append(global_event_index + 1)
                    transition_intervals.append(transition_value)
                    transition_rule_ids.append(transition_rule_id)
                    transition_lookup[(trial_idx, event_idx)] = transition_value
                    transition_rule_lookup[(trial_idx, event_idx)] = transition_rule_id
                    following_transition = transition_value
                    following_transition_rule_id = transition_rule_id
                    next_event_onset = event_offset + post_value + transition_value
                else:
                    next_event_onset = None

                schedule_table.append(
                    {
                        "run_event_index": global_event_index,
                        "trial_id": trial_idx,
                        "trial_index": trial_idx,
                        "trial_template_id": template_id,
                        "trial_type_id": template["trial_type"],
                        "event_index_within_trial": event_idx,
                        "event_category": category,
                        "event_code": code,
                        "trial_start": cursor,
                        "realized_trial_start_interval": trial_start,
                        "event_onset": event_cursor,
                        "realized_event_duration": duration,
                        "event_offset": event_offset,
                        "realized_post_event_interval": post_value,
                        "following_event_transition_interval": following_transition,
                        "following_inter_trial_interval": None,
                        "following_rest_interval": None,
                        "trial_end": None,
                        "event_duration_rule_id": duration_rule_id,
                        "trial_start_rule_id": start_rule_id,
                        "post_event_rule_id": post_rule_id,
                        "event_transition_rule_id": following_transition_rule_id,
                        "inter_trial_rule_id": None,
                        "rest_rule_id": None,
                    }
                )

                if next_event_onset is not None:
                    event_cursor = next_event_onset
                global_event_index += 1

            trial_end_event_index.append(global_event_index - 1)
            trial_end = event_offsets[-1] + post_event_intervals[-1]
            trial_ends.append(trial_end)
            schedule_table[-1]["trial_end"] = trial_end
            cursor = trial_end

            if trial_idx < len(trial_sequence) - 1:
                inter_value, inter_rule_id = self._sample_value(
                    self.inter_trial_interval_spec, None, "inter_trial_interval", rng
                )
                inter_trial_intervals.append(inter_value)
                inter_trial_rule_ids.append(inter_rule_id)
                if (
                    self.rest_every_n_trials
                    and (trial_idx + 1) % self.rest_every_n_trials == 0
                ):
                    rest_value, rest_rule_id = self._sample_value(
                        self.rest_interval_spec, None, "rest_interval", rng
                    )
                else:
                    rest_value, rest_rule_id = 0.0, _rule_id("rest_interval", "none")
                rest_intervals.append(rest_value)
                rest_rule_ids.append(rest_rule_id)
                schedule_table[-1]["following_inter_trial_interval"] = inter_value
                schedule_table[-1]["following_rest_interval"] = rest_value
                schedule_table[-1]["inter_trial_rule_id"] = inter_rule_id
                schedule_table[-1]["rest_rule_id"] = rest_rule_id
                cursor += inter_value + rest_value

        legacy_event_aligned_inter_trial = [0.0] * len(order)
        for boundary_trial_idx, inter_value in enumerate(inter_trial_intervals):
            next_event_index = trial_start_event_index[boundary_trial_idx + 1]
            legacy_event_aligned_inter_trial[next_event_index] = inter_value

        return {
            "order": order,
            "event_categories": event_categories,
            "realized_event_durations": event_durations,
            "trial_ids": trial_ids,
            "event_index_within_trial": event_index_within_trial,
            "trial_template_ids": trial_template_ids,
            "trial_type_ids": trial_type_ids,
            "trial_start_event_index": trial_start_event_index,
            "trial_end_event_index": trial_end_event_index,
            "realized_trial_start_intervals": trial_start_intervals,
            "trial_starts": trial_starts,
            "trial_ends": trial_ends,
            "event_onsets": event_onsets,
            "event_offsets": event_offsets,
            "realized_post_event_intervals": post_event_intervals,
            "within_trial_transition_from_event_index": transition_from,
            "within_trial_transition_to_event_index": transition_to,
            "realized_event_transition_intervals": transition_intervals,
            "inter_trial_boundary_after_trial": list(
                range(max(len(trial_sequence) - 1, 0))
            ),
            "realized_inter_trial_intervals": inter_trial_intervals,
            "realized_rest_intervals": rest_intervals,
            "legacy_event_aligned_inter_trial": legacy_event_aligned_inter_trial,
            "selector_provenance": {
                "event_duration_rule_ids": event_duration_rule_ids,
                "trial_start_rule_ids": trial_start_rule_ids,
                "post_event_rule_ids": post_event_rule_ids,
                "event_transition_rule_ids": transition_rule_ids,
                "inter_trial_rule_ids": inter_trial_rule_ids,
                "rest_rule_ids": rest_rule_ids,
            },
            "schedule_table": schedule_table,
        }

    def event_sequence_from_template_ids(self, template_sequence):
        """Flatten a conceptual-trial template sequence into modeled event codes."""
        sequence = []
        for template_id in template_sequence:
            for event in self.templates_by_id[template_id]["events"]:
                sequence.append(int(event["code"]))
        return sequence


class Optimisation:
    """Run the genetic-algorithm design search loop for a configured experiment.

    Typical usage is ``optimisation.optimise()`` followed by
    ``optimisation.selected_design(0)`` to retrieve the best representative
    design; see ``MIGRATION_2.0.md`` for the full recommended workflow.
    """

    def __init__(
        self,
        experiment: Experiment,
        weights: list[float],
        preruncycles: int,
        cycles: int,
        seed: int | None = None,
        I: int = 4,
        G: int = 20,
        R: list[float] | None = None,
        q: float = 0.01,
        Aoptimality: bool = True,
        folder: str | Path | None = None,
        outdes: int = 3,
        convergence: int | None = 1000,
        max_candidate_attempts: int = 10000,
        optimisation: str = "GA",
    ):
        """Configure an optimisation run over designs sampled from an experiment.

        Parameters
        ----------
        experiment:
            The :class:`Experiment` specification designs are sampled from.
        weights:
            Four-element list ``[Fe_weight, Fd_weight, Ff_weight, Fc_weight]``
            giving the linear combination used for each design's overall
            score ``F``. A metric with weight ``0`` is not computed (except
            ``Ff``/``Fc``, which are always computed).
        preruncycles:
            Number of generations run in each calibration prerun (one for
            ``Fe``, one for ``Fd``) used to estimate ``FeMax``/``FdMax``
            before the main search, when that metric has positive weight.
        cycles:
            Number of generations run in the main search after calibration.
        seed:
            Base seed for this optimisation's RNG streams; defaults to the
            parent experiment's seed if not given.
        I:
            Number of new immigrant designs freshly sampled and injected each
            generation (a diversity mechanism), distributed across
            ``["blocked", "random", "msequence"]`` order types per ``R``.
        G:
            Target population size: the number of designs used to seed the
            initial generation, and the cap each subsequent generation is
            trimmed down to (keeping the highest-scoring designs) after
            mutation, crossover, and immigration.
        R:
            Three-element list of proportions (default ``[0.4, 0.4, 0.2]``)
            controlling the mix of ``["blocked", "random", "msequence"]``
            order-generation strategies used when sampling new candidate
            designs.
        q:
            Mutation rate passed to ``Design.mutation`` when the population is
            not highly correlated; a fixed, larger mutation rate is used
            instead when the population has converged (mean pairwise
            correlation above ``0.6``).
        Aoptimality:
            If ``True``, use A-optimality for ``Fe``/``Fd``; otherwise use
            D-optimality.
        folder:
            Optional output directory for reports and exports.
        outdes:
            Number of representative designs to select via clustering in
            :meth:`evaluate`.
        convergence:
            Patience, in generations, for early stopping: the search stops
            after this many consecutive generations with no strict
            improvement in the generation-best score. ``None`` or ``0``
            disables early stopping.
        max_candidate_attempts:
            Maximum attempts to construct one valid candidate design before
            raising an error (guards against impossible constraints).
        optimisation:
            Search strategy identifier: ``"GA"`` applies mutation, crossover,
            and immigration each generation; ``"simulation"`` applies only
            immigration (pure random resampling, no evolutionary operators).
        """
        self.exp = experiment
        self.weights = weights
        self.preruncycles = preruncycles
        self.cycles = cycles
        self.seed = seed or experiment.seed
        self.I = I
        self.G = G
        self.R = [0.4, 0.4, 0.2] if R is None else R
        self.q = q
        self.Aoptimality = Aoptimality
        self.folder = Path(folder).absolute() if folder else None
        self.outdes = outdes
        self.convergence = convergence
        if self.convergence is not None:
            if not isinstance(self.convergence, int) or self.convergence < 0:
                raise ValueError("convergence must be None or a non-negative integer")
        if not isinstance(max_candidate_attempts, int) or max_candidate_attempts <= 0:
            raise ValueError("max_candidate_attempts must be a positive integer")
        self.max_candidate_attempts = max_candidate_attempts
        self.optimisation = optimisation
        self.designs = []
        self.optima = []
        self.bestdesign = None
        self.cov = None
        self._seed_counter = 0
        self.bestscore = float("-inf")
        self.bestdesign_generation = None
        self.generations_completed = 0
        self.finished = False
        self.stop_reason = None
        self._stagnation_generations = 0
        self._last_candidate_failure = "candidate generation has not been attempted yet"
        self._last_candidate_exception = None

    def _next_rng(self, label: int = 0) -> np.random.Generator:
        """Advance the optimisation RNG stream deterministically."""
        self._seed_counter += 1
        ss = np.random.SeedSequence([self.seed, self._seed_counter, label])
        return np.random.default_rng(ss)

    def change_seed(self):
        """Increment the optimisation seed to start a fresh search trajectory."""
        self.seed = self.seed + 1000 if self.seed < 4 * 10**9 else 1
        return self

    def check_develop(self, design, weights=None):
        """Validate and score a candidate design before keeping it."""
        weights = self.weights if weights is None else weights
        if self.exp.maxrep is not None and not design.check_maxrep(self.exp.maxrep):
            self._last_candidate_failure = "candidate exceeded maxrep"
            self._last_candidate_exception = None
            return False
        if self.exp.hardprob and not design.check_hardprob():
            self._last_candidate_failure = (
                "candidate violated hard probability constraints"
            )
            self._last_candidate_exception = None
            return False
        if len(np.unique(design.order)) < self.exp.n_stimuli:
            self._last_candidate_failure = (
                "candidate omitted one or more stimulus categories"
            )
            self._last_candidate_exception = None
            return False
        out = design.designmatrix()
        if out is False:
            self._last_candidate_failure = "design matrix construction failed"
            self._last_candidate_exception = None
            return False
        if not (
            np.all(np.isfinite(np.asarray(design.Xnonconv)))
            and np.all(np.isfinite(np.asarray(design.Xconv)))
        ):
            self._last_candidate_failure = "design matrices contained non-finite values"
            self._last_candidate_exception = None
            return False
        design.FCalc(
            weights, confoundorder=self.exp.confoundorder, Aoptimality=self.Aoptimality
        )
        component_scores = np.array(
            [design.Fe, design.Fd, design.Ff, design.Fc, design.F], dtype=float
        )
        if not np.all(np.isfinite(component_scores)):
            self._last_candidate_failure = "candidate scores contained non-finite values"
            self._last_candidate_exception = None
            return False
        self._last_candidate_failure = ""
        self._last_candidate_exception = None
        return design

    def _raise_candidate_generation_error(self, attempts, target):
        """Raise a bounded candidate-generation failure with the last known cause."""
        message = (
            f"Failed to produce {target} valid candidate design(s) after {attempts} attempts "
            f"for mode {self.exp.mode!r}. Last failure: {self._last_candidate_failure}."
        )
        if self._last_candidate_exception is not None:
            raise RuntimeError(message) from self._last_candidate_exception
        raise RuntimeError(message)

    def _make_design_from_order(self, order, rng):
        """Create a :class:`Design` from a flat event order."""
        schedule = self.exp.realize_flat_order(order, rng)
        return Design(experiment=self.exp, schedule=schedule, trial_sequence=list(order))

    def _make_design_from_templates(self, template_sequence, rng):
        """Create a :class:`Design` from a conceptual-trial template sequence."""
        schedule = self.exp.realize_from_trial_sequence(
            template_sequence, rng, template_sequence
        )
        return Design(
            experiment=self.exp,
            schedule=schedule,
            trial_sequence=list(template_sequence),
            template_sequence=list(template_sequence),
        )

    def add_new_designs(self, weights=None, R=None):
        """Populate the current generation with newly sampled candidate designs."""
        weights = self.weights if weights is None else weights
        if not R:
            R = np.round(np.array(self.R) * self.G).tolist()
        target = int(np.sum(R))
        NDes = 0
        attempts = 0
        while NDes < target:
            if attempts >= self.max_candidate_attempts:
                self._raise_candidate_generation_error(attempts, target - NDes)
            attempts += 1
            rng = self._next_rng(100 + NDes)
            ind = int(np.sum(NDes >= np.cumsum(R)))
            ordertype = ["blocked", "random", "msequence"][ind]
            try:
                if self.exp.mode == "flat_fixed_order":
                    des = self._make_design_from_order(self.exp.order, rng)
                elif self.exp.mode == "fixed_trials":
                    des = self._make_design_from_templates(
                        self.exp.fixed_trial_sequence, rng
                    )
                elif self.exp.mode == "template_sampled":
                    template_sequence = self.exp.sample_trial_sequence(rng)
                    des = self._make_design_from_templates(template_sequence, rng)
                else:
                    order = generate.order(
                        self.exp.n_stimuli,
                        self.exp.n_trials,
                        self.exp.P.tolist(),
                        ordertype=ordertype,
                        rng=rng,
                    )
                    des = self._make_design_from_order(order, rng)
            except Exception as exc:
                self._last_candidate_failure = (
                    "candidate construction raised an exception"
                )
                self._last_candidate_exception = exc
                continue
            fulldes = self.check_develop(des, weights)
            if fulldes is False:
                continue
            self.designs.append(fulldes)
            NDes += 1
        return self

    def _clean_designs(self, weights):
        """Remove duplicate designs and backfill the population if needed."""
        if len(self.designs) <= 1:
            return self
        if self.exp.mode in {"fixed_trials", "template_sampled"}:
            seen = {}
            keep = []
            for idx, des in enumerate(self.designs):
                key = tuple(des.order)
                if key not in seen:
                    seen[key] = idx
                    keep.append(des)
            removed = len(self.designs) - len(keep)
            self.designs = keep
            if removed > 0:
                self.add_new_designs(R=[0, removed, 0], weights=weights)
            return self
        n = 0
        rm = 0
        while n == 0:
            # np.corrcoef collapses to a bare scalar (not a matrix) when
            # given a single row, so check the population size directly
            # rather than inferring it from the shape of its output --
            # removals below can shrink the population to 1 design before
            # the end-of-function backfill runs.
            if len(self.designs) <= 1:
                n = 1
                continue
            orders = [x.order for x in self.designs]
            cors = np.corrcoef(orders)
            isone = np.isclose(cors, 1.0)
            np.fill_diagonal(isone, 0)
            if np.sum(isone) == 0:
                n = 1
            else:
                ind = np.where(isone)
                remove = ind[1][ind[0] == ind[0][0]]
                self.designs = [
                    des for idx, des in enumerate(self.designs) if idx not in remove
                ]
                rm += len(remove)
        if rm > 0:
            self.add_new_designs(R=[0, rm, 0], weights=weights)
        return self

    @staticmethod
    def _derive_seed(seed, *labels) -> int:
        """Derive a distinct, reproducible sub-seed from ``seed`` and ``labels``.

        ``Design.mutation``/``Design.crossover`` each build their own RNG from
        a single integer ``seed``. Passing the same ``seed`` straight through
        for every individual (or pair) in a generation makes every one of
        them draw an identical random stream, which collapses the intended
        per-individual variation. Salting ``seed`` with a call-specific label
        keeps results reproducible for a given ``seed`` while decorrelating
        different individuals/pairs/operators from one another.
        """
        return int(np.random.SeedSequence([seed, *labels]).generate_state(1)[0])

    def _mutation(self, weights, seed):
        """Apply mutation to the current generation."""
        signals = [x.Xconv for x in self.designs]
        efficiencies = [x.F for x in self.designs]
        cors = self.pearsonr(signals, self.exp.n_stimuli)
        mncor = np.mean(cors)
        for idx in range(len(self.designs)):
            design = self.designs[idx]
            if design.F == np.max(efficiencies):
                offspring = design
            else:
                rate = 0.2 if mncor > 0.6 else self.q
                mutation_seed = self._derive_seed(seed, 1, idx)
                offspring = design.mutation(rate, seed=mutation_seed)
                offspring = self.check_develop(offspring, weights)
            if offspring is not False:
                self.designs[idx] = offspring
        return self

    def _crossover(self, weights, seed):
        """Apply crossover to parent pairs in the current generation."""
        crossind = range(len(self.designs))
        nparents = len(crossind)
        npairs = int(nparents / 2.0)
        rng = np.random.default_rng(seed)
        coupling = rng.choice(nparents, size=(npairs * 2), replace=False)
        coupling = [crossind[x] for x in coupling]
        pairing = [[coupling[i], coupling[i + 1]] for i in np.arange(0, npairs * 2, 2)]
        for pair_idx, couple in enumerate(pairing):
            pair_seed = self._derive_seed(seed, 2, pair_idx)
            baby1, baby2 = self.designs[couple[0]].crossover(
                self.designs[couple[1]], seed=pair_seed
            )
            for baby in [baby1, baby2]:
                baby = self.check_develop(baby, weights)
                if baby is not False:
                    self.designs.append(baby)
        return self

    def _immigration(self, weights, noim):
        """Inject newly sampled designs into the current generation."""
        R = np.ceil(np.array(self.R) * noim).tolist()
        self.add_new_designs(R=R, weights=weights)
        return self

    def to_next_generation(self, weights=None, seed=1234, optimisation=None):
        """Advance one generation of the configured search strategy."""
        weights = self.weights if weights is None else weights
        optimisation = self.optimisation if optimisation is None else optimisation
        if self.exp.mode not in {"flat_fixed_order", "fixed_trials"}:
            self._clean_designs(weights)
            if optimisation == "GA":
                self._mutation(weights, seed)
                self._crossover(weights, seed)
                self._immigration(weights, noim=self.I)
            elif optimisation == "simulation":
                self._immigration(weights, noim=self.I)
        else:
            self._immigration(weights, noim=self.I)

        # Defense in depth: check_develop() and clear() already reject
        # non-finite scores at every entry point, but matrix inversions used
        # by Fe/Fd are sensitive to floating-point rounding, which can differ
        # under multi-threaded BLAS. Never let a non-finite F reach the
        # generation-best/cutoff logic below, where it could crash or
        # silently empty the population.
        finite_designs = [des for des in self.designs if np.isfinite(des.F)]
        dropped = len(self.designs) - len(finite_designs)
        self.designs = finite_designs
        if dropped > 0:
            self.add_new_designs(R=[0, dropped, 0], weights=weights)

        efficiencies = [x.F for x in self.designs]
        maximum = np.max(efficiencies)
        self.optima.append(maximum)
        bestind = [ind for ind, val in enumerate(efficiencies) if val == maximum][0]
        generation_best = self.designs[bestind]
        gen = len(self.optima)
        self.generations_completed = gen
        if self.bestdesign is None or maximum > self.bestscore:
            self.bestscore = float(maximum)
            self.bestdesign = generation_best
            self.bestdesign_generation = gen
            self._stagnation_generations = 0
        else:
            self._stagnation_generations += 1
        convergence_limit = self.convergence
        if convergence_limit in {0, None}:
            self.finished = False
            self.stop_reason = None
        elif self._stagnation_generations >= convergence_limit:
            self.finished = True
            self.stop_reason = (
                "no improvement in generation-best score for "
                f"{convergence_limit} consecutive generation(s)"
            )
        if len(self.designs) > self.G:
            # Index self.G - 1 (not self.G): sorted descending, that's the
            # G-th largest score, so keeping F >= cutoff retains exactly the
            # top G designs. Indexing self.G would keep the (G+1)-th
            # largest as the cutoff, retaining G + 1 designs instead of G.
            cutoff = np.sort(efficiencies)[::-1][self.G - 1]
            self.designs = [des for des in self.designs if des.F >= cutoff]
        return self

    def clear(self, weights=None):
        """Reset the current population while preserving the last best design.

        The preserved design's cached ``F``/``Fe``/``Fd``/``Fc``/``Ff`` were
        computed under whichever weight vector was active in the phase that
        just ended (e.g. a Fe-only or Fd-only calibration prerun). Comparing
        that stale score directly against designs freshly scored under a
        different weight vector is invalid -- component scores live on
        different numeric scales, so a leftover Fe-phase score can spuriously
        outrank every real candidate in a following Fd-phase (or vice versa).
        Rescoring the preserved design under ``weights`` (the vector that
        will govern the upcoming phase, defaulting to ``self.weights``)
        keeps it on the same footing as the rest of the population.
        """
        previous_best = self.bestdesign
        self.designs = []
        self.optima = []
        self.finished = False
        self.stop_reason = None
        self.bestdesign = None
        self.bestscore = float("-inf")
        self.bestdesign_generation = None
        self.generations_completed = 0
        self._stagnation_generations = 0
        if previous_best:
            previous_best.FCalc(
                self.weights if weights is None else weights,
                confoundorder=self.exp.confoundorder,
                Aoptimality=self.Aoptimality,
            )
            component_scores = np.array(
                [
                    previous_best.Fe,
                    previous_best.Fd,
                    previous_best.Ff,
                    previous_best.Fc,
                    previous_best.F,
                ],
                dtype=float,
            )
            # Match check_develop()'s safety guard: a non-finite rescored
            # score (e.g. from an ill-conditioned matrix inversion) must not
            # silently poison the next population -- a NaN/inf F sorts as
            # the maximum and can make every real candidate compare False
            # against it, emptying the population.
            if np.all(np.isfinite(component_scores)):
                self.designs.append(previous_best)
        return self

    def optimise(self):
        """Run the full optimisation procedure, including normalization passes.

        If ``Fc``/``Ff`` are uncalibrated (``FcMax``/``FfMax`` left at ``1``),
        they are calibrated analytically first. If ``Fe``/``Fd`` have positive
        weight and are uncalibrated (``FeMax``/``FdMax`` left at ``1``), a
        short ``preruncycles``-generation prerun optimising that metric alone
        is run to estimate its calibration reference. The main search then
        runs for ``cycles`` generations (or until ``convergence`` triggers
        early stopping).

        Returns
        -------
        Optimisation
            ``self``, with ``designs`` holding the final generation and
            ``bestdesign`` holding the single highest-scoring design found.
            Call :meth:`evaluate` (or :meth:`selected_design`, which calls it
            automatically) to obtain clustered representative outputs.
        """
        if self.exp.FcMax == 1 and self.exp.FfMax == 1:
            self.exp.max_eff()
        if self.exp.FeMax == 1 and self.weights[0] > 0:
            self.clear(weights=[1, 0, 0, 0])
            self.add_new_designs(weights=[1, 0, 0, 0])
            with progress_bar(text="Optimizing") as progress:
                task = progress.add_task(
                    description="optimize", total=len(range(self.preruncycles))
                )
                for _ in range(self.preruncycles):
                    self.to_next_generation(seed=self.seed, weights=[1, 0, 0, 0])
                    progress.update(task, advance=1)
                    if self.finished:
                        break
            self.exp.FeMax = float(np.max(self.bestdesign.F))
        if self.exp.FdMax == 1 and self.weights[1] > 0:
            self.clear(weights=[0, 1, 0, 0])
            self.add_new_designs(weights=[0, 1, 0, 0])
            with progress_bar(text="Optimizing") as progress:
                task = progress.add_task(
                    description="optimize", total=len(range(self.preruncycles))
                )
                for _ in range(self.preruncycles):
                    self.to_next_generation(seed=self.seed, weights=[0, 1, 0, 0])
                    progress.update(task, advance=1)
                    if self.finished:
                        break
            self.exp.FdMax = float(np.max(self.bestdesign.F))
        self.clear()
        self.add_new_designs()
        with progress_bar(text="Optimizing") as progress:
            task = progress.add_task(
                description="optimize", total=len(range(self.cycles))
            )
            for _ in range(self.cycles):
                self.to_next_generation(seed=self.seed)
                progress.update(task, advance=1)
                if self.finished:
                    break
        return self

    def selected_design(self, rank: int = 0):
        """Return one evaluated representative design from the current selected outputs.

        This is the recommended public entry point for retrieving a design
        after :meth:`optimise`; it calls :meth:`evaluate` automatically the
        first time it is needed.

        Parameters
        ----------
        rank:
            Index into the ``outdes`` clustered representative designs
            (0-indexed). Each cluster's representative is the
            highest-scoring design within that cluster; clusters are not
            necessarily ordered by score, so ``rank=0`` is one representative
            design rather than guaranteed to be the single global best (use
            ``bestdesign`` for that).

        Returns
        -------
        Design
            The representative design for the requested cluster rank.
        """
        if self.bestdesign is None or not self.designs:
            raise RuntimeError(
                "selected_design() requires optimise() to run before selecting outputs"
            )
        if not hasattr(self, "out"):
            self.evaluate()
        if rank < 0 or rank >= len(self.out):
            raise IndexError(f"selected design rank {rank} is out of range")
        return self.designs[self.out[rank]]

    def evaluate(self):
        """Cluster final designs and choose representative reported outputs.

        Clusters the final generation's designs into ``outdes`` groups via
        k-means on their convolved design matrices, then within each cluster
        keeps the highest-scoring design as that cluster's representative.
        Reorders ``self.designs`` so cluster representatives are retrievable
        via ``self.out`` (populated here) and :meth:`selected_design`.

        Returns
        -------
        Optimisation
            ``self``, with ``designs`` reordered by cluster, ``out`` holding
            each cluster's representative-design index, ``clus`` holding each
            reordered design's cluster label, and ``cov`` holding the
            pairwise design-signal correlation matrix.
        """
        if self.bestdesign is None or not self.designs:
            raise RuntimeError(
                "evaluate() requires optimise() to run before selecting outputs"
            )
        shape = self.bestdesign.Xconv.shape
        des = np.zeros([np.prod(shape), len(self.designs)])
        efficiencies = np.array([x.F for x in self.designs])
        for d in range(len(self.designs)):
            hrf = []
            for stim in range(shape[1]):
                hrf = hrf + self.designs[d].Xconv[:, stim].tolist()
            des[:, d] = hrf
        clus = sklearn.cluster.k_means(des.T, self.outdes, random_state=self.seed)[1]
        out = []
        new_designs = []
        clusters = []
        first = 0
        for c in range(self.outdes):
            ids = np.where(clus == c)[0]
            id_ordered = ids[np.flipud(np.argsort(efficiencies[ids]))]
            out.append(first)
            for d in id_ordered:
                clusters.append(c)
                new_designs.append(self.designs[d])
                first += 1
        self.designs = new_designs
        self.out = out
        self.clus = clusters
        signals = [x.Xconv for x in self.designs]
        self.cov = self.pearsonr(signals, self.exp.n_stimuli)
        return self

    def download(self):
        """Write report artifacts, schedule exports, and onset files to disk."""
        if not self.folder:
            raise ValueError("No folder defined to download output.")
        if self.cov is None:
            self.evaluate()
        if self.folder.exists():
            files = self.folder.glob("**/design_*")
            for f in files:
                shutil.rmtree(f)
        else:
            self.folder.mkdir(parents=True, exist_ok=True)
        reportfile = "report.pdf"
        report.make_report(self, self.folder / reportfile)
        files = []
        for des in range(self.outdes):
            (self.folder / f"design_{str(des)}").mkdir(parents=True, exist_ok=True)
            design = self.designs[self.out[des]]
            for stim in range(self.exp.n_stimuli):
                onsetsfile = Path(f"design_{str(des)}") / f"stimulus_{str(stim)}.txt"
                onsubsets = [
                    str(x)
                    for x in np.array(design.event_onsets)[np.array(design.order) == stim]
                ]
                with open(self.folder / onsetsfile, "w+") as f:
                    for line in onsubsets:
                        f.write(line)
                        f.write("\n")
                files.append(onsetsfile)
            export_path = self.folder / f"design_{str(des)}" / "event_schedule.json"
            export_path.write_text(
                json.dumps(design.export_payload(), indent=2, default=str),
                encoding="utf-8",
            )
            files.append(Path(f"design_{str(des)}") / "event_schedule.json")
        files.append(Path(reportfile))
        zip_subdir = "OptimalDesign"
        self.zip_filename = f"{zip_subdir}.zip"
        self.file = BytesIO()
        zf = zipfile.ZipFile(self.file, "w")
        for fpath in files:
            zf.write(self.folder / fpath, Path(zip_subdir) / fpath)
        zf.close()
        return self

    @staticmethod
    def pearsonr(signals, nstim):
        """Compute pairwise mean regressor correlations between candidate designs."""
        varcov = np.zeros([len(signals), len(signals)])
        for sig1 in range(len(signals)):
            for sig2 in range(sig1, len(signals)):
                cors = np.diag(
                    np.corrcoef(t(signals[sig1]), t(signals[sig2]))[nstim:, :nstim]
                )
                varcov[sig1, sig2] = np.mean(cors)
                varcov[sig2, sig1] = np.mean(cors)
        return varcov
