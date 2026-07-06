from __future__ import annotations

import tempfile
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from pdfrw import PdfDict, PdfReader, PdfWriter
from pdfrw.buildxobj import pagexobj
from pdfrw.toreportlab import makerl
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def make_report(population, outfile: str | Path = "NeuroDesign.pdf"):
    """Create a report of a finished design optimisation."""
    backend = matplotlib.get_backend()
    plt.switch_backend("agg")

    if not isinstance(population.cov, np.ndarray):
        population.evaluate()

    styles = getSampleStyleSheet()
    logofile = Path(__file__).parent / "media" / "NeuroDes.png"
    temp_paths = []

    title = "NeuroDesign: optimisation report"
    formatted_time = time.ctime()
    ptext = f"Document created: {formatted_time}"
    corr = f"""During the optimisation, the designs are mixed
with each other to find better combinations.
As such, the designs can look very similar.
Actually, the genetic algorithm uses natural selection as a basis,
and as such, the designs can be clustered in families.
This is the design-to-design correlation matrix for the final population of {population.G} designs."""
    designs = f"""The following figure shows in the upper panel the optimisation score
over the {len(population.optima)} generations in the final optimisation run.
Below are the expected signals of the selected representative designs
from different families.
The labels use the final design index after evaluation, not the generation number.
Each row also reports the weighted score and the component metrics (Fe, Fd, Ff, Fc)."""

    def _save_figure(fig):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        path = Path(tmp.name)
        temp_paths.append(path)
        fig.savefig(path, format="png", dpi=150, bbox_inches="tight")
        return path

    def _draw_paragraph(pdf, text, style, x, y, max_width):
        paragraph = Paragraph(text, style)
        _, height = paragraph.wrap(max_width, letter[1])
        paragraph.drawOn(pdf, x, y - height)
        return y - height

    def _styled_table(rows, col_widths):
        table = Table(rows, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.75, "black"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), ["white", "#f5f5f5"]),
                ]
            )
        )
        return table

    corr_fig, corr_ax = plt.subplots(figsize=(6.5, 6))
    corr_im = corr_ax.imshow(
        population.cov, interpolation="nearest", clim=(-1, 1), cmap="RdBu"
    )
    corr_ax.set_title("Correlation matrix of final designs")
    corr_fig.colorbar(corr_im, ax=corr_ax, ticks=[-1, 0, 1])
    corr_path = _save_figure(corr_fig)
    plt.close(corr_fig)

    selected_indices = list(population.out[: population.outdes])
    design_fig = plt.figure(figsize=(12, max(10, 3.1 * population.outdes + 3)))
    gs = gridspec.GridSpec(population.outdes + 4, 5, hspace=1.0, wspace=0.5)
    score_ax = plt.subplot(gs[:2, :])
    generations = np.arange(1, len(population.optima) + 1)
    score_ax.plot(generations, population.optima, color="#1f5a91", lw=2)
    if len(generations) > 0:
        if len(generations) == 1:
            score_ax.set_xlim(0.5, 1.5)
        else:
            score_ax.set_xlim(1, len(generations))
        tick_step = max(1, int(np.ceil(len(generations) / 10)))
        score_ax.set_xticks(generations[::tick_step])
    score_ax.set_xlabel("Generation")
    score_ax.set_ylabel("Best score")
    score_ax.set_title("Optimisation trace")
    score_ax.grid(alpha=0.2)

    for des in range(population.outdes):
        final_idx = selected_indices[des]
        design = population.designs[final_idx]
        stdes = des + 2
        signal_ax = plt.subplot(gs[stdes, :3])
        signal_ax.plot(design.Xconv, lw=1.8)
        signal_ax.set_yticks([])
        signal_ax.set_xticks([])
        cluster_label = (
            population.clus[final_idx] if hasattr(population, "clus") else "NA"
        )
        metric_label = (
            f"Design {final_idx} | cluster {cluster_label} | "
            f"F={design.F:.3f}  Fe={design.Fe:.3f}  Fd={design.Fd:.3f}  "
            f"Ff={design.Ff:.3f}  Fc={design.Fc:.3f}"
        )
        signal_ax.text(
            0.0,
            -0.18,
            metric_label,
            transform=signal_ax.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )
        for spine in signal_ax.spines.values():
            spine.set_visible(False)

        cov_ax = plt.subplot(gs[stdes, 3])
        varcov = np.corrcoef(design.Xconv.T)
        cov_im = cov_ax.imshow(varcov, interpolation="nearest", clim=(-1, 1), cmap="RdBu")
        cov_ax.set_xticks([])
        cov_ax.set_yticks([])
        if des == 0:
            cov_ax.text(
                0.5,
                -0.18,
                "Regressor corr.",
                transform=cov_ax.transAxes,
                fontsize=8,
                ha="center",
                va="top",
            )
        plt.colorbar(cov_im, ax=cov_ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)

        eig_ax = plt.subplot(gs[stdes, 4])
        eigenv = np.diag(np.linalg.svd(design.Xconv.T)[1])
        eig_im = eig_ax.imshow(eigenv, interpolation="nearest", clim=(0, 1))
        eig_ax.set_xticks([])
        eig_ax.set_yticks([])
        if des == 0:
            eig_ax.text(
                0.5,
                -0.18,
                "Singular values",
                transform=eig_ax.transAxes,
                fontsize=8,
                ha="center",
                va="top",
            )
        plt.colorbar(eig_im, ax=eig_ax, ticks=[0, 1], fraction=0.046, pad=0.04)

    design_path = _save_figure(design_fig)
    plt.close(design_fig)

    def _format_value(value):
        if value is None:
            return "None"
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=3, separator=", ")
        if isinstance(value, dict):
            return ", ".join(f"{key}: {_format_value(val)}" for key, val in value.items())
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        return str(value)

    def _table_text(value):
        return Paragraph(_format_value(value), styles["BodyText"])

    probability_text = _table_text(population.exp.P)
    contrast_text = _table_text(population.exp.C)
    weight_text = _table_text(population.weights)
    spec = population.exp.export_specification()

    mode_labels = {
        "flat_generated": "Flat one-event generated order",
        "flat_fixed_order": "Flat one-event fixed order",
        "fixed_trials": "Fixed conceptual-trial sequence",
        "template_sampled": "Probabilistic conceptual-trial templates",
    }
    order_mode = mode_labels.get(population.exp.mode, population.exp.mode)
    order_details = {
        "order": spec["order"],
        "trial_templates": spec["trial_templates"],
        "trials": spec["trials"],
        "trial_template_probabilities": spec["trial_template_probabilities"],
        "n_conceptual_trials": spec["n_conceptual_trials"],
    }

    stim_duration_mode = "Event duration specifications"
    stim_duration_details = spec["event_durations_requested"]
    iti_mode = "Inter-trial interval"
    iti_details = spec["inter_trial_interval_requested"]
    conceptual_trials = spec["n_conceptual_trials"]
    selected_design = (
        population.designs[selected_indices[0]] if selected_indices else None
    )
    modeled_events = (
        len(getattr(selected_design, "order", []))
        if selected_design is not None
        else None
    )

    exp = [
        ["Setting", "Value"],
        ["Repetition time (TR):", _table_text(population.exp.TR)],
        ["Conceptual trial count:", _table_text(conceptual_trials)],
        ["Modeled event count:", _table_text(modeled_events)],
        ["Number of scans:", _table_text(population.exp.n_scans)],
        ["Number of different stimuli:", _table_text(population.exp.n_stimuli)],
        ["Stimulus probabilities:", probability_text],
        ["Order generation:", _table_text(order_mode)],
        ["Order details:", _table_text(order_details)],
        ["Stimulus duration mode:", _table_text(stim_duration_mode)],
        ["Stimulus duration details:", _table_text(stim_duration_details)],
        ["Base stimulus duration (s):", _table_text(population.exp.stim_duration)],
        ["Trial container max (s):", _table_text(population.exp.trial_max)],
        [
            "Trial-start interval spec:",
            _table_text(spec["trial_start_interval_requested"]),
        ],
        ["Post-event interval spec:", _table_text(spec["post_event_interval_requested"])],
        [
            "Event-transition interval spec:",
            _table_text(spec["event_transition_interval_requested"]),
        ],
        ["Expected trial duration (s):", _table_text(population.exp.trial_duration)],
        ["Total experiment duration (s):", _table_text(population.exp.duration)],
        ["Rest cadence (trials):", _table_text(population.exp.rest_every_n_trials)],
        ["Rest interval spec:", _table_text(spec["rest_interval_requested"])],
        ["Contrasts:", contrast_text],
        ["Inter-trial interval mode:", _table_text(iti_mode)],
        ["Inter-trial interval details:", _table_text(iti_details)],
        ["Hard probabilities:", _table_text(population.exp.hardprob)],
        ["Maximum number of repeated stimuli:", _table_text(population.exp.maxrep)],
        ["Resolution of design:", _table_text(population.exp.resolution)],
        ["Assumed autocorrelation:", _table_text(population.exp.rho)],
    ]

    body_story = []
    intro = "Experimental settings"
    body_story.append(Paragraph(intro, styles["Heading2"]))
    body_story.append(Spacer(1, 12))
    body_story.append(_styled_table(exp, [190, 322]))

    optset = "Optimalisation settings"
    body_story.append(Paragraph(optset, styles["Heading2"]))
    body_story.append(Spacer(1, 12))

    opt = [
        ["Setting", "Value"],
        ["Optimisation weights (Fe,Fd,Ff,Fc):", weight_text],
        ["A-optimality?", _table_text(population.Aoptimality)],
        ["Number of designs in each generation:", _table_text(population.G)],
        ["Number of immigrants in each generation:", _table_text(population.I)],
        ["Confounding order:", _table_text(population.exp.confoundorder)],
        [
            "Convergence patience (stagnant generations):",
            _table_text(
                "disabled"
                if population.convergence in {None, 0}
                else population.convergence
            ),
        ],
        ["Completed generations:", _table_text(population.generations_completed)],
        [
            "Stop reason:",
            _table_text(population.stop_reason or "completed planned generations"),
        ],
        ["Number of precycles:", _table_text(population.preruncycles)],
        ["Number of cycles:", _table_text(population.cycles)],
        ["Percentage of mutations:", _table_text(population.q)],
        ["Seed:", _table_text(population.seed)],
    ]

    body_story.append(_styled_table(opt, [190, 322]))

    summary_rows = [["Selection", "Final design", "Cluster", "F", "Fe", "Fd", "Ff", "Fc"]]
    for rank, final_idx in enumerate(selected_indices, start=1):
        design = population.designs[final_idx]
        cluster_label = (
            population.clus[final_idx] if hasattr(population, "clus") else "NA"
        )
        summary_rows.append(
            [
                _table_text(rank),
                _table_text(final_idx),
                _table_text(cluster_label),
                _table_text(f"{design.F:.3f}"),
                _table_text(f"{design.Fe:.3f}"),
                _table_text(f"{design.Fd:.3f}"),
                _table_text(f"{design.Ff:.3f}"),
                _table_text(f"{design.Fc:.3f}"),
            ]
        )

    body_story.append(Paragraph("Selected design summary", styles["Heading2"]))
    body_story.append(Spacer(1, 12))
    body_story.append(
        Paragraph(
            "These are the representative designs shown in the figure. "
            "The final design index refers to the reordered design list "
            "after clustering.",
            styles["BodyText"],
        )
    )
    body_story.append(Spacer(1, 8))
    body_story.append(_styled_table(summary_rows, [55, 70, 55, 42, 42, 42, 42, 42]))

    intro_tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    intro_tmp.close()
    body_tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    body_tmp.close()
    intro_pdf = Path(intro_tmp.name)
    body_pdf = Path(body_tmp.name)
    temp_paths.extend([intro_pdf, body_pdf])

    try:
        page_width, page_height = letter
        margin_x = 40
        top_y = page_height - 40
        usable_width = page_width - (2 * margin_x)

        pdf = canvas.Canvas(str(intro_pdf), pagesize=letter)

        pdf.drawImage(str(logofile), margin_x, top_y - 90, width=72, height=90)
        y = top_y - 102
        y = _draw_paragraph(pdf, title, styles["Title"], margin_x, y, usable_width)
        y -= 12
        y = _draw_paragraph(pdf, ptext, styles["Normal"], margin_x, y, usable_width)
        y -= 16
        y = _draw_paragraph(
            pdf,
            "Correlation between designs",
            styles["Heading2"],
            margin_x,
            y,
            usable_width,
        )
        y -= 12
        y = _draw_paragraph(pdf, corr, styles["Normal"], margin_x, y, usable_width)
        y -= 12
        pdf.drawImage(
            str(corr_path),
            margin_x,
            70,
            width=usable_width,
            height=max(1, y - 82),
            preserveAspectRatio=True,
            anchor="n",
        )
        pdf.showPage()

        y = top_y
        y = _draw_paragraph(
            pdf, "Selected designs", styles["Heading2"], margin_x, y, usable_width
        )
        y -= 12
        y = _draw_paragraph(pdf, designs, styles["Normal"], margin_x, y, usable_width)
        y -= 12
        pdf.drawImage(
            str(design_path),
            margin_x,
            50,
            width=usable_width,
            height=max(1, y - 62),
            preserveAspectRatio=True,
            anchor="n",
        )
        pdf.save()

        body_doc = SimpleDocTemplate(
            str(body_pdf),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=18,
        )
        body_doc.build(body_story)

        writer = PdfWriter()
        writer.addpages(PdfReader(str(intro_pdf)).pages)
        writer.addpages(PdfReader(str(body_pdf)).pages)
        writer.write(str(outfile))
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)

    plt.switch_backend(backend)


def _form_xo_reader(imgdata):
    (page,) = PdfReader(imgdata).pages
    return pagexobj(page)


class PdfImage(Flowable):
    def __init__(self, img_data, width: int = 200, height: int = 200):
        super().__init__()
        self.img_width = width
        self.img_height = height
        self.img_data = img_data

    def wrap(self, availWidth, availHeight):
        return self.img_width, self.img_height

    def draw(self):
        canv = self.canv
        img = self.img_data
        if isinstance(img, PdfDict):
            xscale = self.img_width / float(img.BBox[2])
            yscale = self.img_height / float(img.BBox[3])
            canv.saveState()
            canv.scale(xscale, yscale)
            canv.doForm(makerl(canv, img))
            canv.restoreState()
        else:
            canv.drawImage(img, 0, 0, self.img_width, self.img_height)
