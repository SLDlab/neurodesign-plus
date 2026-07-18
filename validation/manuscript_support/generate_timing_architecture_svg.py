from __future__ import annotations

import os
from pathlib import Path

import matplotlib

# Force a headless backend before any other matplotlib import: this script
# never shows a window, but plt.subplots() otherwise probes for an
# interactive backend (e.g. TkAgg) and can fail on a machine with a broken
# or partial Tk install even though no display is actually needed.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(
    os.environ.get(
        "NEURODESIGN_VALIDATION_SVG_OUTPUT",
        REPO_ROOT / "validation" / "_artifacts" / "timing_svg",
    )
)
DOCS_IMAGES_DIRS = [
    REPO_ROOT / "manuscript" / "docs_images",
    REPO_ROOT / "docs" / "_images",
]

PANEL_A_FORMULA = r"$T_{\mathrm{pre}} + E + T_{\mathrm{post}} + T_{\mathrm{ITI}}$"
PANEL_B_FORMULA = (
    r"$T_{\mathrm{pre}} + \sum_{i=1}^{n-1}(E_i + T_{\mathrm{post},i} + T_{\mathrm{trans},i}) "
    r"+ E_n + T_{\mathrm{post},n} + T_{\mathrm{ITI}}$"
)


def _svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1480" height="980" viewBox="0 0 1480 980">
  <style>
    .title { font: 700 30px 'Arial'; fill: #102A43; }
    .subtitle { font: 700 20px 'Arial'; fill: #243B53; }
    .panel { font: 700 18px 'Arial'; fill: #102A43; }
    .label { font: 600 14px 'Arial'; fill: #102A43; }
    .small { font: 500 13px 'Arial'; fill: #334E68; }
    .tiny { font: 500 12px 'Arial'; fill: #486581; }
    .light { font: 500 12px 'Arial'; fill: #627D98; }
  </style>
  <rect width="1480" height="980" fill="#F8FBFF"/>
  <text x="60" y="60" class="title">Neurodesign-Plus 2.0 timing architecture</text>
  <text x="60" y="94" class="small">Conceptual trials determine sampling and boundaries. Flattened modeled events determine event-level timing, frequency, and transition metrics.</text>

  <rect x="50" y="130" width="1380" height="210" rx="18" fill="#FFFFFF" stroke="#D9E2EC"/>
  <text x="80" y="170" class="panel">Panel A. Original one-event conceptual trial</text>
  <line x1="100" y1="250" x2="1340" y2="250" stroke="#BCCCDC" stroke-width="4"/>
  <rect x="150" y="214" width="155" height="72" fill="#D9E2EC" stroke="#BCCCDC"/>
  <rect x="305" y="214" width="240" height="72" fill="#4F6D7A" stroke="#BCCCDC"/>
  <rect x="545" y="214" width="180" height="72" fill="#F4F1DE" stroke="#BCCCDC"/>
  <rect x="725" y="214" width="235" height="72" fill="#CBD5E0" stroke="#BCCCDC"/>
  <text x="180" y="256" class="label">pre-trial interval</text>
  <text x="390" y="256" class="label" fill="#FFFFFF">stimulus event</text>
  <text x="575" y="256" class="label">post-event interval</text>
  <text x="742" y="256" class="label">inter-trial interval</text>
  <text x="150" y="205" class="tiny">trial starts</text>
  <text x="720" y="205" class="tiny">trial ends</text>
  <text x="150" y="312" class="small">Flattened metric axis:</text>
  <text x="332" y="312" class="small"><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">pre</tspan><tspan dy="-5"> + </tspan><tspan font-style="italic">E</tspan><tspan> + </tspan><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">post</tspan><tspan dy="-5"> + </tspan><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">ITI</tspan></text>
  <text x="150" y="333" class="light">Original one-event trials expose one modeled event per conceptual trial. <tspan font-style="italic">T</tspan><tspan dy="3" font-size="80%">pre</tspan><tspan dy="-3"> = pre-trial interval, </tspan><tspan font-style="italic">E</tspan><tspan> = event, </tspan><tspan font-style="italic">T</tspan><tspan dy="3" font-size="80%">post</tspan><tspan dy="-3"> = post-event interval, </tspan><tspan font-style="italic">T</tspan><tspan dy="3" font-size="80%">ITI</tspan><tspan dy="-3"> = inter-trial interval.</tspan></text>

  <rect x="50" y="370" width="1380" height="260" rx="18" fill="#FFFFFF" stroke="#D9E2EC"/>
  <text x="80" y="410" class="panel">Panel B. Version-2 multi-event conceptual trial</text>
  <line x1="100" y1="505" x2="1340" y2="505" stroke="#BCCCDC" stroke-width="4"/>
  <rect x="145" y="470" width="145" height="70" fill="#D9E2EC" stroke="#BCCCDC"/>
  <rect x="290" y="470" width="120" height="70" fill="#4F6D7A" stroke="#BCCCDC"/>
  <rect x="410" y="470" width="105" height="70" fill="#F4F1DE" stroke="#BCCCDC"/>
  <rect x="515" y="470" width="115" height="70" fill="#EAD2AC" stroke="#BCCCDC"/>
  <rect x="630" y="470" width="150" height="70" fill="#6C5B7B" stroke="#BCCCDC"/>
  <rect x="780" y="470" width="105" height="70" fill="#F4F1DE" stroke="#BCCCDC"/>
  <rect x="885" y="470" width="115" height="70" fill="#EAD2AC" stroke="#BCCCDC"/>
  <rect x="1000" y="470" width="150" height="70" fill="#F8B195" stroke="#BCCCDC"/>
  <rect x="1150" y="470" width="105" height="70" fill="#F4F1DE" stroke="#BCCCDC"/>
  <rect x="1255" y="470" width="120" height="70" fill="#CBD5E0" stroke="#BCCCDC"/>
  <text x="168" y="512" class="label">pre-trial interval</text>
  <text x="335" y="512" class="label" fill="#FFFFFF">event 1</text>
  <text x="433" y="512" class="label">post 1</text>
  <text x="532" y="512" class="label">1 to 2</text>
  <text x="680" y="512" class="label" fill="#FFFFFF">event 2</text>
  <text x="803" y="512" class="label">post 2</text>
  <text x="902" y="512" class="label">2 to n</text>
  <text x="1050" y="512" class="label">event n</text>
  <text x="1174" y="512" class="label">post n</text>
  <text x="1270" y="512" class="label">inter-trial</text>
  <text x="148" y="565" class="small">Flattened metric axis:</text>
  <text x="310" y="565" class="small"><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">pre</tspan><tspan dy="-5"> + </tspan><tspan font-size="130%">&#931;</tspan><tspan dy="6" font-size="60%">i=1</tspan><tspan dy="-14" font-size="60%">n-1</tspan><tspan dy="8">(</tspan><tspan font-style="italic">E</tspan><tspan dy="5" font-size="70%">i</tspan><tspan dy="-5"> + </tspan><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">post,i</tspan><tspan dy="-5"> + </tspan><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">trans,i</tspan><tspan dy="-5">) + </tspan><tspan font-style="italic">E</tspan><tspan dy="5" font-size="70%">n</tspan><tspan dy="-5"> + </tspan><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">post,n</tspan><tspan dy="-5"> + </tspan><tspan font-style="italic">T</tspan><tspan dy="5" font-size="70%">ITI</tspan></text>
  <text x="148" y="590" class="light">Conceptual-trial count T governs sampling and boundaries; modeled-event count E governs Ff and Fc. <tspan font-style="italic">T</tspan><tspan dy="3" font-size="80%">trans</tspan><tspan dy="-3"> = within-trial transition.</tspan></text>
  <text x="148" y="611" class="light">Optional rest intervals are inserted only at conceptual-trial boundaries and do not create modeled events.</text>

  <rect x="50" y="660" width="1380" height="270" rx="18" fill="#FFFFFF" stroke="#D9E2EC"/>
  <text x="80" y="700" class="panel">Panel C. Three Case 10 conceptual-trial schedules</text>
  <text x="85" y="745" class="label">standard</text>
  <rect x="215" y="720" width="130" height="50" fill="#4F6D7A" stroke="#BCCCDC"/><text x="248" y="751" class="label" fill="#FFFFFF">cue_easy</text>
  <rect x="345" y="720" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="358" y="751" class="label">transition</text>
  <rect x="435" y="720" width="145" height="50" fill="#6C5B7B" stroke="#BCCCDC"/><text x="462" y="751" class="label" fill="#FFFFFF">choice_left</text>
  <rect x="580" y="720" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="593" y="751" class="label">transition</text>
  <rect x="670" y="720" width="125" height="50" fill="#F8B195" stroke="#BCCCDC"/><text x="700" y="751" class="label">feedback</text>
  <rect x="795" y="720" width="120" height="50" fill="#CBD5E0" stroke="#BCCCDC"/><text x="817" y="751" class="label">inter-trial</text>

  <text x="85" y="820" class="label">hint_branch</text>
  <rect x="215" y="795" width="130" height="50" fill="#C06C84" stroke="#BCCCDC"/><text x="248" y="826" class="label" fill="#FFFFFF">cue_hard</text>
  <rect x="345" y="795" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="358" y="826" class="label">transition</text>
  <rect x="435" y="795" width="105" height="50" fill="#99B898" stroke="#BCCCDC"/><text x="468" y="826" class="label">hint</text>
  <rect x="540" y="795" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="553" y="826" class="label">transition</text>
  <rect x="630" y="795" width="150" height="50" fill="#355C7D" stroke="#BCCCDC"/><text x="658" y="826" class="label" fill="#FFFFFF">choice_right</text>
  <rect x="780" y="795" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="793" y="826" class="label">transition</text>
  <rect x="870" y="795" width="125" height="50" fill="#F8B195" stroke="#BCCCDC"/><text x="900" y="826" class="label">feedback</text>
  <rect x="995" y="795" width="120" height="50" fill="#CBD5E0" stroke="#BCCCDC"/><text x="1017" y="826" class="label">inter-trial</text>

  <text x="85" y="895" class="label">hold_branch</text>
  <rect x="215" y="870" width="130" height="50" fill="#4F6D7A" stroke="#BCCCDC"/><text x="248" y="901" class="label" fill="#FFFFFF">cue_easy</text>
  <rect x="345" y="870" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="358" y="901" class="label">transition</text>
  <rect x="435" y="870" width="105" height="50" fill="#E84A5F" stroke="#BCCCDC"/><text x="470" y="901" class="label">hold</text>
  <rect x="540" y="870" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="553" y="901" class="label">transition</text>
  <rect x="630" y="870" width="145" height="50" fill="#6C5B7B" stroke="#BCCCDC"/><text x="657" y="901" class="label" fill="#FFFFFF">choice_left</text>
  <rect x="775" y="870" width="90" height="50" fill="#EAD2AC" stroke="#BCCCDC"/><text x="788" y="901" class="label">transition</text>
  <rect x="865" y="870" width="125" height="50" fill="#F8B195" stroke="#BCCCDC"/><text x="895" y="901" class="label">feedback</text>
  <rect x="990" y="870" width="120" height="50" fill="#CBD5E0" stroke="#BCCCDC"/><text x="1012" y="901" class="label">inter-trial</text>
</svg>
"""


def _render_png(target: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.8, 9.8))
    ax.set_xlim(0, 1480)
    ax.set_ylim(980, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1480, 980, facecolor="#F8FBFF"))
    ax.text(
        60,
        60,
        "Neurodesign-Plus 2.0 timing architecture",
        fontsize=22,
        fontweight="bold",
        color="#102A43",
    )
    ax.text(
        60,
        94,
        "Conceptual trials determine sampling and boundaries. Flattened modeled events determine event-level timing and Ff/Fc.",
        fontsize=11,
        color="#334E68",
    )
    for x, y, w, h in [(50, 130, 1380, 210), (50, 370, 1380, 260), (50, 660, 1380, 270)]:
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor="white", edgecolor="#D9E2EC", linewidth=1.5)
        )
    ax.text(
        80,
        170,
        "Panel A. Original one-event conceptual trial",
        fontsize=16,
        fontweight="bold",
        color="#102A43",
    )
    ax.text(
        80,
        410,
        "Panel B. Version-2 multi-event conceptual trial",
        fontsize=16,
        fontweight="bold",
        color="#102A43",
    )
    ax.text(
        80,
        700,
        "Panel C. Three Case 10 conceptual-trial schedules",
        fontsize=16,
        fontweight="bold",
        color="#102A43",
    )

    panel_a = [
        (150, 214, 155, "#D9E2EC"),
        (305, 214, 240, "#4F6D7A"),
        (545, 214, 180, "#F4F1DE"),
        (725, 214, 235, "#CBD5E0"),
    ]
    panel_b = [
        (145, 470, 145, "#D9E2EC"),
        (290, 470, 120, "#4F6D7A"),
        (410, 470, 105, "#F4F1DE"),
        (515, 470, 115, "#EAD2AC"),
        (630, 470, 150, "#6C5B7B"),
        (780, 470, 105, "#F4F1DE"),
        (885, 470, 115, "#EAD2AC"),
        (1000, 470, 150, "#F8B195"),
        (1150, 470, 105, "#F4F1DE"),
        (1255, 470, 120, "#CBD5E0"),
    ]
    for x, y, width, color in panel_a + panel_b:
        ax.add_patch(
            Rectangle(
                (x, y), width, 70, facecolor=color, edgecolor="#BCCCDC", linewidth=1.0
            )
        )
    panel_a_labels = [
        (228, 252, "pre-trial interval", "#102A43"),
        (425, 252, "stimulus event", "white"),
        (635, 252, "post-event interval", "#102A43"),
        (842, 252, "inter-trial interval", "#102A43"),
    ]
    for x, y, text, color in panel_a_labels:
        ax.text(
            x,
            y,
            text,
            fontsize=12,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
    ax.text(150, 205, "trial starts", fontsize=10, color="#486581")
    ax.text(720, 205, "trial ends", fontsize=10, color="#486581")
    ax.text(150, 312, PANEL_A_FORMULA, fontsize=11, color="#334E68")
    ax.text(
        150,
        332,
        r"Original one-event trials expose one modeled event per conceptual trial. "
        r"$T_{\mathrm{pre}}$ = pre-trial interval, $E$ = event, $T_{\mathrm{post}}$ = "
        r"post-event interval, $T_{\mathrm{ITI}}$ = inter-trial interval.",
        fontsize=10,
        color="#627D98",
    )

    panel_b_labels = [
        (218, 505, "pre-trial interval", "#102A43"),
        (350, 505, "event 1", "white"),
        (462, 505, "post 1", "#102A43"),
        (572, 505, "1 to 2", "#102A43"),
        (705, 505, "event 2", "white"),
        (833, 505, "post 2", "#102A43"),
        (943, 505, "2 to n", "#102A43"),
        (1075, 505, "event n", "#102A43"),
        (1202, 505, "post n", "#102A43"),
        (1296, 505, "inter-trial", "#102A43"),
    ]
    for x, y, text, color in panel_b_labels:
        ax.text(
            x,
            y,
            text,
            fontsize=11,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
        )
    ax.text(148, 565, PANEL_B_FORMULA, fontsize=10.2, color="#334E68")
    ax.text(
        148,
        588,
        r"Conceptual-trial count T governs sampling and boundaries; modeled-event count E "
        r"governs Ff and Fc. $T_{\mathrm{trans}}$ = within-trial transition.",
        fontsize=10,
        color="#627D98",
    )
    ax.text(
        148,
        608,
        "Optional rest intervals are inserted only at conceptual-trial boundaries and do not create modeled events.",
        fontsize=10,
        color="#627D98",
    )

    trial_rows = [
        (
            720,
            "standard",
            [
                (215, 130, "#4F6D7A"),
                (345, 90, "#EAD2AC"),
                (435, 145, "#6C5B7B"),
                (580, 90, "#EAD2AC"),
                (670, 125, "#F8B195"),
                (795, 120, "#CBD5E0"),
            ],
        ),
        (
            795,
            "hint_branch",
            [
                (215, 130, "#C06C84"),
                (345, 90, "#EAD2AC"),
                (435, 105, "#99B898"),
                (540, 90, "#EAD2AC"),
                (630, 150, "#355C7D"),
                (780, 90, "#EAD2AC"),
                (870, 125, "#F8B195"),
                (995, 120, "#CBD5E0"),
            ],
        ),
        (
            870,
            "hold_branch",
            [
                (215, 130, "#4F6D7A"),
                (345, 90, "#EAD2AC"),
                (435, 105, "#E84A5F"),
                (540, 90, "#EAD2AC"),
                (630, 145, "#6C5B7B"),
                (775, 90, "#EAD2AC"),
                (865, 125, "#F8B195"),
                (990, 120, "#CBD5E0"),
            ],
        ),
    ]
    trial_text = [
        (
            720,
            [
                "cue_easy",
                "transition",
                "choice_left",
                "transition",
                "feedback",
                "inter-trial",
            ],
        ),
        (
            795,
            [
                "cue_hard",
                "transition",
                "hint",
                "transition",
                "choice_right",
                "transition",
                "feedback",
                "inter-trial",
            ],
        ),
        (
            870,
            [
                "cue_easy",
                "transition",
                "hold",
                "transition",
                "choice_left",
                "transition",
                "feedback",
                "inter-trial",
            ],
        ),
    ]
    for y, label, segments in trial_rows:
        ax.text(85, y + 30, label, fontsize=13, fontweight="bold", color="#102A43")
        for x, width, color in segments:
            ax.add_patch(
                Rectangle(
                    (x, y), width, 50, facecolor=color, edgecolor="#BCCCDC", linewidth=1.0
                )
            )
    for (y, labels), (_, _, segments) in zip(trial_text, trial_rows):
        for (x, width, color), text in zip(segments, labels):
            txt_color = (
                "white"
                if color in {"#4F6D7A", "#C06C84", "#355C7D", "#6C5B7B"}
                else "#102A43"
            )
            ax.text(
                x + width / 2,
                y + 30,
                text,
                fontsize=10,
                fontweight="bold",
                color=txt_color,
                ha="center",
                va="center",
            )
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def main() -> None:
    """Write the timing-architecture SVG and rendered PNG outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = OUTPUT_DIR / "architecture_schematic.svg"
    png_path = OUTPUT_DIR / "architecture_schematic.png"
    svg_path.write_text(_svg(), encoding="utf-8")
    _render_png(png_path)
    for docs_images_dir in DOCS_IMAGES_DIRS:
        docs_images_dir.mkdir(parents=True, exist_ok=True)
        (docs_images_dir / svg_path.name).write_text(
            svg_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (docs_images_dir / png_path.name).write_bytes(png_path.read_bytes())


if __name__ == "__main__":
    main()
