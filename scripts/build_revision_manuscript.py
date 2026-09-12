"""Build the result-gated one-column revision-v2 Word manuscript.

The builder refuses to create a submission manuscript until frozen results and
external physical cross-checks exist. It never imports legacy result files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "manuscript" / "revision_v2" / "Handover_Aware_DRL_TN_NTN.docx"


def require_final_inputs(artifacts: Path):
    required = [
        ROOT / "data" / "canonical_v2" / "manifest.json",
        ROOT / "standards" / "external_reference_outputs.csv",
        artifacts / "conformance" / "reference_results.csv",
        artifacts / "development" / "ppo_critic" / "selected_configuration.json",
        artifacts / "evaluation" / "paired_test_results.csv",
        artifacts / "evaluation" / "seed_level_summary.csv",
        artifacts / "evaluation" / "seed_aware_statistics.csv",
        artifacts / "geometry" / "paired_geometry_results.csv",
        artifacts / "geometry" / "geometry_difficulty.csv",
        artifacts / "sensitivity" / "raw_kpi_pareto.csv",
        ROOT / "manuscript" / "revision_v2" / "figures" / "figure1_pipeline.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("manuscript build blocked until frozen inputs exist: " + "; ".join(missing))
    manifest = json.loads((ROOT / "data" / "canonical_v2" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("manuscript build blocked: canonical dataset is incomplete")


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def set_cell_border(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{side}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:color"), "D9D9D9")
        borders.append(edge)
    tc_pr.append(borders)


def native_equation(document: Document, text: str):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    math_para = OxmlElement("m:oMathPara")
    math = OxmlElement("m:oMath")
    run = OxmlElement("m:r")
    run_props = OxmlElement("m:rPr")
    math_style = OxmlElement("m:sty")
    math_style.set(qn("m:val"), "p")
    run_props.append(math_style)
    run.append(run_props)
    value = OxmlElement("m:t")
    value.text = text
    run.append(value)
    math.append(run)
    math_para.append(math)
    paragraph._p.append(math_para)
    return paragraph


def add_caption(document: Document, label: str, number: int, text: str):
    caption = document.add_paragraph(style="Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.add_run(f"{label} {number}. {text}")


def add_summary_table(document: Document, summary: pd.DataFrame):
    selected = summary[summary["policy"].isin(["random_valid", "sinr_greedy", "hysteresis", "cql", "ppo", "dqn", "oracle"])]
    mean = selected.groupby("policy", as_index=False).mean(numeric_only=True)
    order = ["random_valid", "sinr_greedy", "hysteresis", "cql", "ppo", "dqn", "oracle"]
    mean["order"] = mean["policy"].map({name: index for index, name in enumerate(order)})
    mean = mean.sort_values("order")
    table = document.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    headers = ["Policy", "Reference utility", "Throughput Mbps", "Latency ms", "Handovers"]
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        set_cell_shading(cell, "1F4E78")
        set_cell_border(cell)
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.font.bold = True
    for row_index, row in enumerate(mean.itertuples(), start=1):
        cells = table.add_row().cells
        values = [
            row.policy, f"{row.reference_utility:.2f}", f"{row.throughput_mbps:.2f}",
            f"{row.latency_ms:.2f}", f"{row.switches:.2f}",
        ]
        for index, value in enumerate(values):
            cells[index].text = value
            set_cell_border(cells[index])
            if row_index % 2 == 0:
                set_cell_shading(cells[index], "F3F7FA")
    add_caption(document, "Table", 1, "Final-test performance averaged over bands; learned-policy summaries retain training seed as the outer unit.")


def build(artifacts: Path, output: Path):
    require_final_inputs(artifacts)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.85); section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.9); section.right_margin = Inches(0.9)
    section.page_width = Inches(8.5); section.page_height = Inches(11)
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"; normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    for style_name, size in [("Title", 16), ("Heading 1", 13), ("Heading 2", 11)]:
        style = document.styles[style_name]
        style.font.name = "Times New Roman"; style.font.size = Pt(size); style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
    if "Caption" not in document.styles:
        document.styles.add_style("Caption", WD_STYLE_TYPE.PARAGRAPH)
    document.styles["Caption"].font.name = "Times New Roman"
    document.styles["Caption"].font.size = Pt(9)
    document.styles["Caption"].font.italic = True

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Handover Aware Deep Reinforcement Learning for Network Selection in Simulated Hybrid TN NTN Systems")
    authors = document.add_paragraph("Anonymous authors for review")
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("Abstract", level=1)
    document.add_paragraph(
        "This study evaluates handover-aware network selection in a standards-informed simulated hybrid terrestrial and non-terrestrial system. "
        "A frozen bank of matched 10-second trajectories supplies identical realized candidate outcomes to all evaluated policies. "
        "Offline conservative Q-learning is evaluated independently from online PPO and DQN. Results are reported with a fixed reference utility, raw link-quality indicators, a finite-horizon clairvoyant oracle, and seed-aware paired inference. "
        "The conclusions are restricted to the validated synthetic environment."
    )
    document.add_paragraph("Keywords: network selection; handover; deep reinforcement learning; non-terrestrial networks; simulation")

    document.add_heading("1 Introduction", level=1)
    document.add_paragraph(
        "Hybrid TN-NTN systems can expose a user equipment to heterogeneous terrestrial, aerial, satellite, and local-access candidates whose availability and link conditions evolve together. "
        "The central methodological requirement is a paired comparison: every policy must select from the same realized candidate set at each decision epoch."
    )
    document.add_paragraph(
        "The contribution is a reproducible simulation study rather than a field-deployment claim. It separates offline CQL from online PPO/DQN, treats Ku, Ka, and S as equally complete band configurations, and uses an oracle only to quantify the headroom remaining under realized episode knowledge."
    )

    document.add_heading("2 Simulator and Scenario Bank", level=1)
    document.add_paragraph(
        "The link simulator implements selected channel-model components from 3GPP TR 38.901, TR 36.777, and TR 38.811 within their stated parameter ranges. ITU-R models provide atmospheric attenuation, while custom mobility, interference, and network-selection components are documented separately. At the system level, the simulator is described as standards-informed; it is not described as 3GPP-compliant."
    )
    document.add_paragraph(
        "The canonical bank contains 1,200 trajectories split by trajectory identity into 800 training, 200 validation, and 200 final-test trajectories. Every trajectory has 60 ten-second steps and every scenario-step-band group contains exactly five candidate records. Unavailable outcomes are null and accompanied by an explicit availability flag."
    )
    native_equation(document, "Log BER = -log10(BER + 10^-12)")
    document.add_paragraph(
        "Interference degradation is clipped at zero. Channel SINR and the hardware-impairment power are combined in the linear domain. KPI generation uses a documented MCS table, spectral-efficiency cap, distance floor, and saturation behavior."
    )

    document.add_heading("3 Learning and Evaluation", level=1)
    document.add_paragraph(
        "CQL observes 21 values: six area indicators and five [available, RSSI-normalized, SINR-normalized] triples. PPO and DQN use those values plus five previous-network indicators, for 26 values. Normalization limits are fitted once on the complete training split across all bands."
    )
    document.add_paragraph(
        "An unavailable action receives reward -1.0, advances time, and leaves the previous valid connection unchanged. The fixed reporting utility balances throughput, latency, and reliability and applies a reference handover penalty of 0.15. A3-inspired hysteresis is tuned only on validation trajectories."
    )
    figure = ROOT / "manuscript" / "revision_v2" / "figures" / "figure1_pipeline.png"
    document.add_picture(str(figure), width=Inches(6.2))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "Figure", 1, "Independent offline CQL and online PPO/DQN branches over the frozen matched scenario bank.")
    document.add_paragraph(
        "The finite-horizon oracle is solved by dynamic programming over the realized full episode, including the handover penalty. It is an upper bound with privileged future knowledge and must dominate each evaluated policy on each paired scenario."
    )

    document.add_heading("4 Results", level=1)
    document.add_paragraph(
        "All policies were evaluated on the same final-test trajectories. Learning curves, stability diagnostics, raw KPIs, policy regret relative to the oracle, and geometry-difficulty measures are retained in frozen result files. The reported inference resamples training seeds first and shared scenarios second."
    )
    summary = pd.read_csv(artifacts / "evaluation" / "seed_level_summary.csv")
    add_summary_table(document, summary)
    statistics = pd.read_csv(artifacts / "evaluation" / "seed_aware_statistics.csv")
    document.add_paragraph(
        f"The seed-aware analysis contains {len(statistics)} policy-comparison and metric summaries with 10,000 hierarchical bootstrap replicates and exact sign-flip tests at the seed level."
    )
    document.add_heading("5 Robustness and Limitations", level=1)
    document.add_paragraph(
        "All trained online seeds are tested under nominal geometry, 350-450 km LEO shells, 900-1,200 km shells, and low-elevation passes capped at 10-30 degrees. Reported difficulty includes oracle-normalized regret, oracle utility, valid-network count, per-tier availability, and outage exposure."
    )
    document.add_paragraph(
        "The study does not establish real-time deployment readiness, MAC-protocol performance, field performance, or network-standard compliance. It does not include the separate quantum/federated extension."
    )
    document.add_heading("6 Reproducibility", level=1)
    document.add_paragraph(
        "Every result is bound to the dataset version, trajectory split, seed, hyperparameters, dependency versions, configuration hash, and result path. The release workflow validates the schema, environment semantics, physical reference vectors, external simulator cross-checks, oracle dominance, and clean-clone reproduction before publication."
    )
    document.add_heading("References", level=1)
    for reference in [
        "3GPP TR 38.901. Study on channel model for frequencies from 0.5 to 100 GHz. https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/",
        "3GPP TR 36.777. Enhanced LTE support for aerial vehicles. https://www.3gpp.org/ftp/Specs/archive/36_series/36.777/",
        "3GPP TR 38.811. Study on New Radio to support non-terrestrial networks. https://www.3gpp.org/ftp/Specs/archive/38_series/38.811/",
        "ITU-R P.838-3. Specific attenuation model for rain for use in prediction methods. https://www.itu.int/rec/R-REC-P.838-3-200503-I/en",
        "ITU-R P.676-13. Attenuation by atmospheric gases and related effects. https://www.itu.int/rec/R-REC-P.676-13-202208-I/en",
        "Düe et al. OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations, 2025. https://github.com/ant-uni-bremen/OpenNTN",
        "Fontes Pupo et al. 5G Link-Level Simulator for Multicast/Broadcast Services, 2023. doi:10.1109/BMSB58369.2023.10211507. LLSim5G repository: https://github.com/EFontesP90/LLSim5G",
    ]:
        document.add_paragraph(reference)
    document.core_properties.title = "Handover Aware Deep Reinforcement Learning for Network Selection in Simulated Hybrid TN NTN Systems"
    document.core_properties.subject = "Revision v2 manuscript"
    document.core_properties.author = "TN NTN revision v2"
    document.save(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "revision_v2")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    build(args.artifacts, args.output)
