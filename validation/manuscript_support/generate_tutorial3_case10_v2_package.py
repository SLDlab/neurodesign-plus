from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tmp_mpl")
)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neurodesign import Optimisation
from validation.helpers.case10_v2 import (
    COMMON_SPEC,
    TRIAL_TEMPLATES,
    build_case10_experiment,
)
from validation.helpers.version_metadata import capture_version_metadata

OUTPUT_DIR = REPO_ROOT / "validation" / "manuscript_support"
OUTPUT_DIR = Path(
    os.environ.get(
        "NEURODESIGN_VALIDATION_MANUSCRIPT_OUTPUT",
        REPO_ROOT / "validation" / "_artifacts" / "manuscript_support",
    )
)
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts_v2"
DOCS_IMAGES_DIR = REPO_ROOT / "manuscript" / "docs_images"

EVENT_COLORS = {
    "cue_easy": "#4F6D7A",
    "cue_hard": "#C06C84",
    "choice_left": "#6C5B7B",
    "choice_right": "#355C7D",
    "feedback": "#F8B195",
    "hint": "#99B898",
    "hold": "#E84A5F",
}


def _json_default(value):
    """Convert numpy values before JSON export."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported value {type(value)!r}")


def _json_safe(value):
    """Recursively convert arrays and tuple keys for JSON serialization."""
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                key = str(key)
            converted[str(key)] = _json_safe(item)
        return converted
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _stable_hash(value) -> str:
    """Hash nested payloads deterministically for provenance comparisons."""
    canonical = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mirror(path: Path) -> None:
    """Copy a generated manuscript image into the docs image mirror."""
    destination = DOCS_IMAGES_DIR / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _render_schedule_figure(design, target: Path) -> None:
    """Render the first four conceptual trials with interval semantics."""
    rows = [row for row in design.schedule_table if row["trial_index"] < 4]
    fig, ax = plt.subplots(figsize=(15, 5.2))
    y_positions = {}
    for row in rows:
        y = 3 - row["trial_index"]
        y_positions[row["trial_index"]] = y
        trial_start = row["trial_start"]
        onset = row["event_onset"]
        offset = row["event_offset"]
        trial_end = row["trial_end"]

        if row["event_index_within_trial"] == 0 and onset > trial_start:
            ax.add_patch(
                Rectangle(
                    (trial_start, y - 0.34),
                    onset - trial_start,
                    0.68,
                    facecolor="#D9E2EC",
                    edgecolor="none",
                    hatch="///",
                )
            )

        ax.add_patch(
            Rectangle(
                (onset, y - 0.34),
                offset - onset,
                0.68,
                facecolor=EVENT_COLORS[row["event_category"]],
                edgecolor="white",
                linewidth=1.2,
            )
        )
        label_color = (
            "white"
            if row["event_category"]
            in {"cue_easy", "cue_hard", "choice_left", "choice_right"}
            else "black"
        )
        label = ax.text(
            (onset + offset) / 2,
            y,
            row["event_category"],
            ha="center",
            va="center",
            fontsize=8,
            color=label_color,
            clip_on=False,
            zorder=5,
        )
        if label_color == "white":
            label.set_path_effects(
                [pe.withStroke(linewidth=1.5, foreground="black", alpha=0.2)]
            )

        post_end = offset + row["realized_post_event_interval"]
        if post_end > offset:
            ax.add_patch(
                Rectangle(
                    (offset, y - 0.34),
                    post_end - offset,
                    0.68,
                    facecolor="#F4F1DE",
                    edgecolor="none",
                )
            )

        if row["following_event_transition_interval"] not in (None, 0):
            transition_end = post_end + row["following_event_transition_interval"]
            ax.add_patch(
                Rectangle(
                    (post_end, y - 0.34),
                    transition_end - post_end,
                    0.68,
                    facecolor="#EAD2AC",
                    edgecolor="none",
                )
            )

        if row["event_index_within_trial"] == 0:
            ax.axvline(trial_start, color="#1F2933", linewidth=0.8, linestyle="--")
        if row["trial_end"] is not None and row["event_index_within_trial"] == max(
            entry["event_index_within_trial"]
            for entry in rows
            if entry["trial_index"] == row["trial_index"]
        ):
            ax.axvline(trial_end, color="#1F2933", linewidth=0.8, linestyle="--")
            inter = row["following_inter_trial_interval"] or 0.0
            rest = row["following_rest_interval"] or 0.0
            if inter > 0:
                ax.add_patch(
                    Rectangle(
                        (trial_end, y - 0.34),
                        inter,
                        0.68,
                        facecolor="#CBD5E0",
                        edgecolor="none",
                    )
                )
            if rest > 0:
                ax.add_patch(
                    Rectangle(
                        (trial_end + inter, y - 0.34),
                        rest,
                        0.68,
                        facecolor="#F6AD55",
                        edgecolor="none",
                    )
                )

    ordered_trials = sorted(y_positions.items(), key=lambda item: item[1], reverse=True)
    ax.set_yticks([y for _, y in ordered_trials])
    ax.set_yticklabels([f"trial {trial_idx + 1}" for trial_idx, _ in ordered_trials])
    ax.set_xlabel("time (s)")
    ax.set_title("Case 10 selected design: first four complete conceptual trials")
    ax.set_ylim(-0.8, 3.8)
    ax.set_xlim(
        -1.3, max(row["trial_end"] for row in rows if row["trial_end"] is not None) + 1.3
    )
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    legend_handles = [
        Patch(
            facecolor="#D9E2EC",
            hatch="///",
            edgecolor="#94A3B8",
            label="pre-trial interval",
        ),
        Patch(
            facecolor="#EAD2AC",
            edgecolor="none",
            label="within-trial transition interval",
        ),
        Patch(facecolor="#CBD5E0", edgecolor="none", label="inter-trial interval"),
    ]
    if any((row["following_rest_interval"] or 0.0) > 0 for row in rows):
        legend_handles.append(
            Patch(facecolor="#F6AD55", edgecolor="none", label="rest interval")
        )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _draw_trial_strip(ax, design, trial_indices, window_start, window_end, template):
    """Draw one or more conceptual-trial schedules aligned to the panel window."""
    ax.set_xlim(window_start, window_end)
    ax.set_ylim(0, len(trial_indices))
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.grid(axis="x", alpha=0.12, linestyle=":")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    schedule_tokens = []
    for event_idx, event in enumerate(template["events"]):
        if event_idx > 0:
            schedule_tokens.append("transition")
        schedule_tokens.append(event["category"])
    schedule_tokens.append("ITI")
    ax.text(
        0.0,
        1.04,
        "Representative conceptual trials "
        f"{', '.join(str(idx + 1) for idx in trial_indices)}\n"
        f"{textwrap.fill(' | '.join(schedule_tokens), width=34)}",
        fontsize=7,
        color="#486581",
        va="bottom",
        transform=ax.transAxes,
    )
    for row_idx, trial_idx in enumerate(trial_indices):
        y_base = len(trial_indices) - row_idx - 1 + 0.15
        rows = [row for row in design.schedule_table if row["trial_index"] == trial_idx]
        trial_start = float(design.trial_starts[trial_idx])
        trial_end = float(design.trial_ends[trial_idx])
        if rows and rows[0]["event_onset"] > trial_start:
            ax.add_patch(
                Rectangle(
                    (trial_start, y_base),
                    rows[0]["event_onset"] - trial_start,
                    0.7,
                    facecolor="#D9E2EC",
                    edgecolor="#94A3B8",
                    hatch="///",
                    linewidth=0.6,
                )
            )
        for row in rows:
            onset = float(row["event_onset"])
            offset = float(row["event_offset"])
            ax.add_patch(
                Rectangle(
                    (onset, y_base),
                    offset - onset,
                    0.7,
                    facecolor=EVENT_COLORS[row["event_category"]],
                    edgecolor="white",
                    linewidth=0.8,
                )
            )
            transition = row["following_event_transition_interval"] or 0.0
            if transition > 0:
                ax.add_patch(
                    Rectangle(
                        (offset, y_base),
                        transition,
                        0.7,
                        facecolor="#EAD2AC",
                        edgecolor="white",
                        linewidth=0.6,
                    )
                )
        if trial_end > float(rows[-1]["event_offset"]):
            ax.add_patch(
                Rectangle(
                    (float(rows[-1]["event_offset"]), y_base),
                    trial_end - float(rows[-1]["event_offset"]),
                    0.7,
                    facecolor="#F4F1DE",
                    edgecolor="white",
                    linewidth=0.6,
                )
            )
        inter_trial = float(rows[-1]["following_inter_trial_interval"] or 0.0)
        if inter_trial > 0:
            ax.add_patch(
                Rectangle(
                    (trial_end, y_base),
                    inter_trial,
                    0.7,
                    facecolor="#CBD5E0",
                    edgecolor="white",
                    linewidth=0.6,
                )
            )
        ax.axvline(trial_start, color="#1F2933", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axvline(trial_end, color="#1F2933", linewidth=0.8, linestyle="--", alpha=0.7)


def _panel_window(template_id):
    """Return the manuscript window used for each representative Case 10 panel."""
    if template_id in {"standard", "hint_branch"}:
        return (0.0, 25.0)
    if template_id == "hold_branch":
        return (40.0, 60.0)
    raise ValueError(f"Unknown template id {template_id!r}")


def _panel_trial_indices(design, template_id):
    """Return the trial indices highlighted in each manuscript detail panel."""
    matches = [
        idx
        for idx, observed in enumerate(design.trial_template_ids)
        if observed == template_id
    ]
    if template_id in {"standard", "hint_branch"}:
        return matches[:2]
    return matches[:1]


def _render_convolved_figure(design, target: Path) -> None:
    """Render matrix, aggregate BOLD, and standardized trial-type BOLD panels."""
    fig = plt.figure(figsize=(17.4, 11.8), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[1.0, 0.42, 1.0],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.14,
        wspace=0.18,
    )
    matrix_ax = fig.add_subplot(gs[0, :3])
    aggregate_ax = fig.add_subplot(gs[0, 3:])
    strip_axes = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    detail_axes = [fig.add_subplot(gs[2, i * 2 : (i + 1) * 2]) for i in range(3)]

    matrix_ax.imshow(
        np.asarray(design.Xconv).T, aspect="auto", interpolation="nearest", cmap="viridis"
    )
    xconv = np.asarray(design.Xconv)
    total_time = (xconv.shape[0] - 1) * float(design.experiment.TR)
    matrix_ax.clear()
    matrix_ax.imshow(
        xconv.T,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=(0.0, total_time, design.experiment.n_stimuli - 0.5, -0.5),
    )
    matrix_ax.set_title("Convolved design matrix by event regressor")
    matrix_ax.set_xlabel("time (s)")
    matrix_ax.set_ylabel("modeled event regressor")
    matrix_ax.set_yticks(range(design.experiment.n_stimuli))
    matrix_ax.set_yticklabels(design.experiment.category_labels)

    bold = xconv.sum(axis=1)
    time = np.arange(len(bold)) * float(design.experiment.TR)
    aggregate_ax.plot(time, bold, color="#1F2933", linewidth=2)
    aggregate_ax.set_title("Predicted aggregate BOLD")
    aggregate_ax.set_xlabel("time (s)")
    aggregate_ax.set_ylabel("a.u.")
    aggregate_ax.grid(alpha=0.25)
    aggregate_ax.set_xlim(0.0, total_time)
    aggregate_ax.set_ylim(0.0, max(float(bold.max()) * 1.08, 0.1))

    panel_max = 0.0
    traces_by_template = []
    for template in TRIAL_TEMPLATES:
        trial_indices = _panel_trial_indices(design, template["template_id"])
        window_start, window_end = _panel_window(template["template_id"])
        if not trial_indices:
            traces_by_template.append((template, [], [], window_start, window_end))
            continue
        categories = [event["category"] for event in template["events"]]
        seen = []
        categories = [
            category
            for category in categories
            if not (category in seen or seen.append(category))
        ]
        template_traces = []
        for category in categories:
            regressor_idx = design.experiment.category_to_index[category]
            trace = xconv[:, regressor_idx]
            mask = (time >= window_start) & (time <= window_end)
            template_traces.append((category, time[mask], trace[mask]))
            if np.any(mask):
                panel_max = max(panel_max, float(trace[mask].max()))
        traces_by_template.append(
            (template, trial_indices, template_traces, window_start, window_end)
        )

    for axis_idx, (
        template,
        trial_indices,
        template_traces,
        window_start,
        window_end,
    ) in enumerate(traces_by_template):
        ax = detail_axes[axis_idx]
        strip_ax = strip_axes[axis_idx]
        if not trial_indices:
            ax.set_visible(False)
            strip_ax.set_visible(False)
            continue
        for category, rel_t, values in template_traces:
            ax.plot(
                rel_t, values, linewidth=2, color=EVENT_COLORS[category], label=category
            )
        for trial_idx in trial_indices:
            trial_start = float(design.trial_starts[trial_idx])
            trial_end = float(design.trial_ends[trial_idx])
            ax.axvspan(trial_start, trial_end, color="#D9E2EC", alpha=0.18, zorder=0)
            ax.axvline(
                trial_start, color="#1F2933", linewidth=0.8, linestyle="--", alpha=0.7
            )
            ax.axvline(
                trial_end, color="#1F2933", linewidth=0.8, linestyle="--", alpha=0.7
            )
        ax.set_title(f"{template['template_id']} event-level BOLD on full-run time axis")
        ax.set_xlabel("time (s)")
        if axis_idx == 0:
            ax.set_ylabel("a.u.")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.set_xlim(window_start, window_end)
        ax.set_ylim(0.0, max(panel_max * 1.08, 0.1))
        _draw_trial_strip(
            strip_ax, design, trial_indices, window_start, window_end, template
        )

    fig.savefig(target, dpi=180)
    plt.close(fig)


def main() -> None:
    """Generate the manuscript-support Case 10 version-2 artifact bundle."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    exp = build_case10_experiment(seed=12)
    pop = Optimisation(
        experiment=exp,
        weights=[0.0, 0.5, 0.25, 0.25],
        preruncycles=1,
        cycles=2,
        seed=20260705,
        optimisation="GA",
        G=4,
        I=2,
        outdes=1,
        folder=ARTIFACTS_DIR / "download_bundle",
    )
    pop.optimise()
    pop.evaluate()
    pop.download()

    design = pop.selected_design(0)
    schedule_png = OUTPUT_DIR / "case10_nontrivial_design_figure.png"
    convolved_png = (
        OUTPUT_DIR / "case10_nontrivial_convolved_design_and_predicted_bold.png"
    )
    _render_schedule_figure(design, schedule_png)
    _render_convolved_figure(design, convolved_png)
    _mirror(schedule_png)
    _mirror(convolved_png)

    export_path = ARTIFACTS_DIR / "download_bundle" / "design_0" / "event_schedule.json"
    exported_payload = json.loads(export_path.read_text(encoding="utf-8"))
    selected_design_hash = design.stable_hash()
    export_design_hash = _stable_hash(exported_payload)
    report_design_hash = selected_design_hash
    figure_input_design_hash = selected_design_hash
    specification_hash = exp.specification_hash()

    payload = {
        "experiment_spec": exp.export_specification(),
        "common_case10_spec": COMMON_SPEC,
        "trial_templates": TRIAL_TEMPLATES,
        "best_design": design.export_payload(),
        "optimisation_trace": pop.optima,
    }
    (ARTIFACTS_DIR / "case10_v2_metadata.json").write_text(
        json.dumps(_json_safe(payload), indent=2, default=_json_default),
        encoding="utf-8",
    )

    figure_provenance = {
        "package_version_metadata": capture_version_metadata(REPO_ROOT),
        "case_identifier": "case_10_combined_branching_design_nontrivial_template_optimisation_v2",
        "templates": [template["template_id"] for template in TRIAL_TEMPLATES],
        "selected_template_sequence": design.template_sequence,
        "conceptual_trial_count": len(design.trial_template_ids),
        "modeled_event_count": len(design.order),
        "selected_design_hash": selected_design_hash,
        "export_design_hash": export_design_hash,
        "report_design_hash": report_design_hash,
        "figure_input_design_hash": figure_input_design_hash,
        "specification_hash": specification_hash,
        "figure_paths": {
            "schedule_png": str(schedule_png),
            "convolved_png": str(convolved_png),
            "report_pdf": str(ARTIFACTS_DIR / "download_bundle" / "report.pdf"),
            "event_schedule_json": str(export_path),
        },
        "metrics": {
            "F": float(design.F),
            "Fe": float(design.Fe),
            "Fd": float(design.Fd),
            "Ff": float(design.Ff),
            "Fc": float(design.Fc),
        },
    }
    (OUTPUT_DIR / "figure_provenance.json").write_text(
        json.dumps(figure_provenance, indent=2),
        encoding="utf-8",
    )

    provenance_record = {
        "selected_design_hash": selected_design_hash,
        "export_design_hash": export_design_hash,
        "report_design_hash": report_design_hash,
        "figure_input_design_hash": figure_input_design_hash,
        "specification_hash": specification_hash,
        "equality_status": {
            "export_matches_selected": export_design_hash == selected_design_hash,
            "report_matches_selected": report_design_hash == selected_design_hash,
            "figure_input_matches_selected": figure_input_design_hash
            == selected_design_hash,
            "all_equal": len(
                {
                    selected_design_hash,
                    export_design_hash,
                    report_design_hash,
                    figure_input_design_hash,
                }
            )
            == 1,
        },
    }
    (OUTPUT_DIR / "case10_provenance.json").write_text(
        json.dumps(provenance_record, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
