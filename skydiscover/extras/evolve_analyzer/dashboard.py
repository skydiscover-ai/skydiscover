"""
dashboard.py
------------
Streamlit dashboard that renders an EvolveLoopReport.
UI matches the EvoExp Analyzer mockup: sidebar navigation, styled panels.

Invoke as:
    streamlit run dashboard.py -- --report-dir ./evolve_analysis_output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import numpy as np
import streamlit as st


st.set_page_config(
    page_title="EvoExp Analyzer",
    page_icon="🧬",
    layout="wide",
)

CUSTOM_CSS = """
<style>
[data-testid="stHeader"] {display:none;}
[data-testid="stToolbar"] {display:none;}
.block-container {padding-top:1.5rem;}

.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:20px}
.stat-card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px 20px}
.stat-card .label{font-size:11px;color:#5f6368;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.stat-card .value{font-size:26px;font-weight:700;color:#202124}
.stat-card .sub{font-size:11px;color:#9aa0a6;margin-top:4px}

.meta-strip{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:12px 20px;display:flex;gap:32px;margin-bottom:20px;flex-wrap:wrap}
.meta-strip .meta-item .mk{font-size:10px;color:#9aa0a6;text-transform:uppercase;letter-spacing:.5px}
.meta-strip .meta-item .mv{font-size:13px;font-weight:600;color:#202124;margin-top:2px}

.exec-box{background:#e8f0fe;border:1px solid #c5d8f7;border-radius:8px;padding:16px 20px;margin-bottom:20px}
.exec-box h3{font-size:13px;font-weight:700;color:#1a73e8;margin-bottom:8px}
.exec-box p{font-size:13px;color:#202124;line-height:1.6;margin-bottom:8px}

.section-title{font-size:14px;font-weight:700;color:#202124;margin-bottom:12px;margin-top:8px}
.section-subtitle{font-size:13px;font-weight:600;color:#3c4043;margin-bottom:8px;margin-top:4px;margin-left:4px}
.section-sub{font-size:12px;color:#5f6368;margin-bottom:16px;margin-top:-8px}

.dim-table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e0e0e0}
.dim-table th{background:#f8f9fa;font-size:11px;font-weight:700;color:#5f6368;text-transform:uppercase;letter-spacing:.5px;padding:10px 14px;text-align:left;border-bottom:1px solid #e0e0e0}
.dim-table td{padding:10px 14px;font-size:13px;border-bottom:1px solid #f1f3f4;vertical-align:middle}
.dim-table tr:last-child td{border-bottom:none}
.dim-table tr:hover td{background:#f8f9fa}
.stars{color:#f9ab00;letter-spacing:1px;font-size:14px}
.stars .empty{color:#e0e0e0}

.badge-crit{background:#fce8e6;color:#c5221f;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;border:1px solid #f4b8b3;text-transform:uppercase;display:inline-block}
.badge-warn{background:#fef7e0;color:#b06000;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;border:1px solid #fdd663;text-transform:uppercase;display:inline-block}
.badge-ok{background:#e6f4ea;color:#137333;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;border:1px solid #81c995;text-transform:uppercase;display:inline-block}
.badge-info{background:#e8f0fe;color:#1a55c0;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px;border:1px solid #a8c5fa;text-transform:uppercase;display:inline-block}

.llm-block{background:#f8f9fa;border-left:3px solid #dadce0;border-radius:0 4px 4px 0;padding:12px 14px;font-size:12px;color:#3c4043;line-height:1.6;margin:10px 0}
.rec-box{border:1.5px solid #1a73e8;border-radius:6px;padding:10px 14px;font-size:12px;color:#202124;line-height:1.6;margin:10px 0}
.rec-box strong{color:#1a73e8}
.root-cause-badge{display:inline-block;background:#fff3e0;color:#e65100;border:1px solid #ffcc80;border-radius:4px;padding:3px 10px;font-size:11px;font-weight:700;letter-spacing:.5px;margin:10px 0}

.seq-table{width:100%;border-collapse:collapse;font-size:12px;margin:12px 0}
.seq-table th{background:#f8f9fa;padding:6px 10px;text-align:left;color:#5f6368;border-bottom:1px solid #e0e0e0;font-weight:600}
.seq-table td{padding:6px 10px;border-bottom:1px solid #f1f3f4}

.data-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:16px}
.data-table th{background:#f8f9fa;padding:8px 12px;text-align:left;color:#5f6368;border-bottom:1px solid #e0e0e0;font-weight:600}
.data-table td{padding:8px 12px;border-bottom:1px solid #f1f3f4}

.pill-row{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;margin-bottom:14px}
.pill{background:#f1f3f4;border-radius:20px;padding:6px 14px;font-size:12px;color:#5f6368;border:1px solid #e0e0e0}
.pill strong{color:#202124}

.insight-box{background:#e8f0fe;border-left:4px solid #1a73e8;border-radius:0 6px 6px 0;padding:12px 16px;font-size:13px;color:#202124;line-height:1.6;margin-top:14px;margin-bottom:14px}
.insight-box strong{color:#1a73e8}

.heat-green{background:#e6f4ea;color:#137333}
.heat-yellow{background:#fef7e0;color:#8a6400}
.heat-red{background:#fce8e6;color:#c5221f}
.heat-neutral{background:#f8f9fa;color:#5f6368}

.cross-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}
.cross-table th,.cross-table td{padding:8px 12px;border:1px solid #e0e0e0;text-align:center}
.cross-table th{background:#f8f9fa;font-weight:600;color:#5f6368}
.cross-table .row-head{text-align:left;font-weight:600;background:#f8f9fa}

.metric-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px;margin-bottom:16px}
.metric-card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px 16px}
.metric-card .m-label{font-size:11px;color:#5f6368;margin-bottom:4px}
.metric-card .m-val{font-size:18px;font-weight:700;color:#202124}
.metric-card .m-sub{font-size:10px;color:#9aa0a6;margin-top:3px}

.panel-card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px 18px}
.panel-card h4{font-size:12px;font-weight:700;color:#5f6368;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}

.artifact-card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px 18px;margin-bottom:12px}
.artifact-freq-bar{height:8px;border-radius:4px;background:#1a73e8;margin-bottom:4px}
.freq-label{font-size:11px;color:#9aa0a6}
.count-badge{background:#e8f0fe;color:#1a55c0;font-size:11px;font-weight:700;padding:2px 8px;border-radius:10px}

.obs-section{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px 18px;margin-top:8px}
.obs-section h4{font-size:12px;font-weight:700;color:#5f6368;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px}
.obs-section ul{padding-left:18px}
.obs-section li{font-size:13px;color:#3c4043;margin-bottom:6px;line-height:1.5}

.budget-bar-wrap{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:20px;margin-bottom:20px}
.stacked-bar{display:flex;height:40px;border-radius:6px;overflow:hidden;margin-bottom:12px}
.stacked-bar .seg{display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;white-space:nowrap}
.budget-legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px}
.budget-legend-item{display:flex;align-items:center;gap:6px;font-size:12px}
.color-sq{width:12px;height:12px;border-radius:2px;flex-shrink:0;display:inline-block}

.status-bar{background:#137333;color:#fff;padding:10px 20px;border-radius:8px;display:flex;align-items:center;gap:14px;margin-bottom:20px;font-size:13px;flex-wrap:wrap}
.status-bar .sb-item{display:flex;flex-direction:column}
.status-bar .sb-label{font-size:9px;opacity:.7;text-transform:uppercase;letter-spacing:.5px}
.status-bar .sb-val{font-size:13px;font-weight:700}
</style>
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_report(report_dir: str) -> dict:
    path = Path(report_dir) / "report.json"
    if not path.exists():
        st.error(f"report.json not found in '{report_dir}'")
        st.stop()
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_df(report_dir: str) -> Optional[pd.DataFrame]:
    base = Path(report_dir)
    for p in [base / "df.parquet", base / "df.csv"]:
        if p.exists():
            try:
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            except (OSError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value, decimals: int = 4) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
        if np.isnan(f):
            return "—"
        return f"{f:.{decimals}g}"
    except (TypeError, ValueError):
        return str(value)


def _stars(rating) -> str:
    if rating is None:
        return '<span class="stars" style="color:#9aa0a6">N/A</span>'
    filled = "●" * rating
    empty = '<span class="empty">●</span>' * (5 - rating)
    return f'<span class="stars">{filled}{empty}</span>'


def _badge(severity: str) -> str:
    s = (severity or "").lower()
    if s == "critical":
        return '<span class="badge-crit">CRITICAL</span>'
    if s in ("high", "warning", "warn"):
        return '<span class="badge-warn">WARN</span>'
    if s in ("ok", "good", "low"):
        return '<span class="badge-ok">GOOD</span>'
    return '<span class="badge-info">INFO</span>'


def _rating_badge(rating) -> str:
    if rating is None:
        return '<span class="badge-info">N/A</span>'
    if rating >= 4:
        return '<span class="badge-ok">GOOD</span>'
    if rating == 3:
        return '<span class="badge-warn">WARN</span>'
    if rating == 2:
        return '<span class="badge-warn">POOR</span>'
    return '<span class="badge-crit">CRITICAL</span>'


def _heat(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "heat-neutral"
    v = float(val)
    if v >= 60:
        return "heat-green"
    if v >= 40:
        return "heat-yellow"
    return "heat-red"


# ---------------------------------------------------------------------------
# Panel: Overview
# ---------------------------------------------------------------------------

def _render_overview(report: dict, df: Optional[pd.DataFrame]) -> None:
    agg = report.get("aggregate_stats", {})
    stag_periods = report.get("stagnation_periods", [])

    # --- Executive Summary (top) ---
    exec_summary_md = report.get("executive_summary_md") or report.get("executive_summary", "")
    if exec_summary_md:
        st.markdown('<div class="exec-box">', unsafe_allow_html=True)
        st.markdown("### Executive Summary")
        st.markdown(exec_summary_md)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Aggregate Stats ---
    st.markdown('<div class="section-title">Aggregate Stats</div>', unsafe_allow_html=True)

    total_iters = agg.get("total_iterations", len(df) if df is not None else "—")
    n_eval = agg.get("n_evaluations", agg.get("total_iterations", "—"))
    n_succ = agg.get("n_successful", "—")
    n_stag_periods = agg.get("n_stagnation_periods", len(stag_periods))
    total_stag_iters = agg.get("total_stagnation_iterations", "—")
    best_score = agg.get("best_score", "—")
    final_score = agg.get("final_score")
    worst_score = agg.get("worst_score", "—")
    duration_h = agg.get("total_duration_hours")

    agg_rows = [
        ("Total iterations", total_iters),
        ("Evaluations", n_eval),
        ("Successful evaluations", n_succ),
        ("Stagnation periods", n_stag_periods),
        ("Stagnation iterations", total_stag_iters),
        ("Best score", _fmt(best_score, 6) if best_score != "—" else "—"),
        ("Final score", _fmt(final_score, 6) if final_score is not None else "N/A"),
        ("Worst score", _fmt(worst_score, 6) if worst_score != "—" else "—"),
        ("Total duration", f"{duration_h:.2f} h" if duration_h is not None else "—"),
    ]
    agg_rows_html = "".join(
        f"<tr><td style='font-weight:600;color:#5f6368;padding:7px 14px;border-bottom:1px solid #f1f3f4'>{k}</td>"
        f"<td style='padding:7px 14px;border-bottom:1px solid #f1f3f4;font-weight:600;color:#202124'>{v}</td></tr>"
        for k, v in agg_rows
    )
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;margin-bottom:20px">
      <table style="width:100%;border-collapse:collapse">
        <tbody>{agg_rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    # --- Experiment Analysis Parameters ---
    st.markdown('<div class="section-title">Experiment Analysis Parameters</div>', unsafe_allow_html=True)

    run_source = report.get("run_source", "—")
    run_config = report.get("run_config_path", "—")
    run_path = report.get("run_path", "—")

    param_items = [
        ("--source", run_source),
        ("--config", run_config),
        ("--path", run_path),
    ]
    param_html = "".join(
        f"<li style='font-size:13px;color:#3c4043;margin-bottom:6px;line-height:1.5'>"
        f"<strong style='color:#202124'>{k}:</strong> "
        f"<code style='background:#f1f3f4;padding:2px 6px;border-radius:3px;font-size:12px'>{v}</code></li>"
        for k, v in param_items
        if v and v != "—"
    )
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px 18px;margin-bottom:20px">
      <ul style="padding-left:18px;margin:0">{param_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # --- LLM Judge (sub-section of Experiment Analysis Parameters) ---
    llm = report.get("llm_judge_status", {}) or {}
    llm_model = llm.get("model", "—")
    llm_provider = llm.get("provider", "—")
    llm_base_url = llm.get("base_url", "")
    llm_status = llm.get("status", "—")
    llm_error = llm.get("error")

    if llm_model != "—":
        st.markdown('<div class="section-subtitle">LLM Judge</div>', unsafe_allow_html=True)
        status_icon = "✅ Connected successfully" if llm_status == "success" else f"❌ {llm_status}"
        model_label = f"{llm_model} ({llm_provider}" + (f" via {llm_base_url}" if llm_base_url else "") + ")"

        judge_items = [("Model", model_label), ("Status", status_icon)]
        if llm_error:
            judge_items.append(("Error", llm_error))
        judge_html = "".join(
            f"<li style='font-size:13px;color:#3c4043;margin-bottom:6px;line-height:1.5'>"
            f"<strong style='color:#202124'>{k}:</strong> {v}</li>"
            for k, v in judge_items
        )
        st.markdown(f"""
    <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px 18px;margin-bottom:20px">
      <ul style="padding-left:18px;margin:0">{judge_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Experiment Setup ---
    algo_name = report.get("algorithm_name") or (run_source if run_source != "—" else None)
    algo_class = report.get("algorithm_class")
    num_islands = report.get("num_islands")
    if algo_name or algo_class:
        st.markdown('<div class="section-title">Experiment Setup</div>', unsafe_allow_html=True)
        setup_items = []
        if algo_name:
            setup_items.append(("Algorithm name", algo_name))
        if algo_class:
            setup_items.append(("Algorithm class", algo_class))
        if num_islands is not None:
            setup_items.append(("Islands", num_islands))
        setup_html = "".join(
            f"<li style='font-size:13px;color:#3c4043;margin-bottom:6px;line-height:1.5'>"
            f"<strong style='color:#202124'>{k}:</strong> "
            f"<code style='background:#f1f3f4;padding:2px 6px;border-radius:3px;font-size:12px'>{v}</code></li>"
            for k, v in setup_items
        )
        st.markdown(f"""
    <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:14px 18px;margin-bottom:20px">
      <ul style="padding-left:18px;margin:0">{setup_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # --- Dimension Overview ---
    dimensions = report.get("dimensions", [])
    if dimensions:
        st.markdown('<div class="section-title">Dimension Overview</div>', unsafe_allow_html=True)
        rows = ""
        for dim in dimensions:
            name = dim.get("name", "Unknown")
            rating = dim.get("rating")
            summary = dim.get("summary", "")
            hl = 'style="background:#fff5f5"' if rating is not None and rating <= 1 else ""
            target_panel = DIMENSION_TO_PANEL.get(name)
            if target_panel:
                name_cell = f'<a href="?panel={quote(target_panel)}" style="color:#1a73e8;text-decoration:none;font-weight:600" target="_self">{name}</a>'
            else:
                name_cell = f"<strong>{name}</strong>"
            rows += f"""<tr {hl}>
              <td>{name_cell}</td>
              <td>{_stars(rating)}</td>
              <td>{_rating_badge(rating)}</td>
              <td style="font-size:12px;color:#5f6368">{summary[:110]}{"…" if len(summary) > 110 else ""}</td>
            </tr>"""
        st.markdown(f"""
        <table class="dim-table">
          <thead><tr><th>Dimension</th><th>Rating</th><th>Status</th><th>Summary</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.info("No dimension data available.")


# ---------------------------------------------------------------------------
# Panel: Score Progression
# ---------------------------------------------------------------------------

def _render_score_progression(report: dict, df: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go

    stag_periods = report.get("stagnation_periods", [])

    if df is not None and "child_score" in df.columns:
        work_df = df.copy()
        if "iteration" not in work_df.columns:
            work_df["iteration"] = range(len(work_df))
        if "best_so_far" not in work_df.columns:
            work_df["best_so_far"] = work_df["child_score"].expanding().max()

        fig = go.Figure()

        stagnation_positions = ["top left", "top right"]
        label_idx = 0
        for p in stag_periods:
            x0 = p.get("start_iteration", 0)
            x1 = p.get("end_iteration") or work_df["iteration"].max()
            sev = p.get("severity", "")
            fill = "rgba(217,48,37,0.08)" if sev == "critical" else "rgba(249,171,0,0.10)"
            border = "rgba(217,48,37,0.3)" if sev == "critical" else "rgba(249,171,0,0.4)"
            show_label = (x1 - x0) > 1
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_color=border,
                          line_dash="dash",
                          annotation_text=f"Stagnation ({sev})<br>{x0}–{x1}" if show_label else "",
                          annotation_position=stagnation_positions[label_idx % 2] if show_label else "top left",
                          annotation_font_size=9)
            if show_label:
                label_idx += 1

        window = max(1, len(work_df) // 10)
        rolling = work_df["child_score"].rolling(window=window, min_periods=1).mean()

        # Color the worst-score point red within the individual scores trace
        worst_idx = work_df["child_score"].idxmin()
        worst_iter = work_df.loc[worst_idx, "iteration"]
        worst_val = work_df.loc[worst_idx, "child_score"]
        marker_colors = ["#d93025" if it == worst_iter else "#8ab4f8"
                         for it in work_df["iteration"]]
        marker_sizes = [10 if it == worst_iter else 5 for it in work_df["iteration"]]

        fig.add_trace(go.Scatter(x=work_df["iteration"], y=work_df["child_score"],
                                  mode="markers", name="Individual Score",
                                  marker=dict(color=marker_colors, size=marker_sizes, opacity=0.8)))
        fig.add_trace(go.Scatter(x=work_df["iteration"], y=rolling,
                                  mode="lines", name="Rolling Mean",
                                  line=dict(color="#9aa0a6", width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=work_df["iteration"], y=work_df["best_so_far"],
                                  mode="lines", name="Best-so-far",
                                  line=dict(color="#1a73e8", width=2.5)))

        # Clip y-axis to the non-outlier range so sentinel scores don't collapse
        # the visible range. Use 2nd–100th percentile of individual scores with
        # 10% padding; keep autorange off so the range is respected.
        finite_scores = work_df["child_score"].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if len(finite_scores) > 0:
            y_low = float(np.percentile(finite_scores, 2))
            y_high = float(np.percentile(finite_scores, 100))
            padding = abs(y_high - y_low) * 0.10 or abs(y_high) * 0.10 or 1.0
            y_min = y_low - padding
            y_max = y_high + padding
        else:
            y_min, y_max = None, None

        # Annotate the worst score; if it falls outside the clipped range, anchor
        # the arrow tip to the bottom of the visible area instead of data coords.
        ann_y = y_min if (y_min is not None and worst_val < y_min) else worst_val
        ann_ay = 40 if (y_min is not None and worst_val < y_min) else -30
        fig.add_annotation(
            x=worst_iter, y=ann_y,
            text=f"Worst: {worst_val:.6f}",
            showarrow=True, arrowhead=2, arrowcolor="#d93025",
            font=dict(color="#d93025", size=10),
            ax=40, ay=ann_ay,
        )

        fig.update_layout(title="Score Progression — All Iterations",
                          xaxis_title="Iteration", yaxis_title="Score",
                          plot_bgcolor="#fff", paper_bgcolor="#fff",
                          legend=dict(orientation="h", y=-0.25),
                          margin=dict(l=40, r=20, t=40, b=60), height=370)
        fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
        if y_min is not None:
            fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0",
                             range=[y_min, y_max], autorange=False)
        else:
            fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No iteration DataFrame available for score progression chart.")

    conv_dim = next((d for d in report.get("dimensions", []) if d.get("name") == "Convergence"), None)
    pills = []
    if conv_dim:
        for ev in conv_dim.get("evidence", [])[:3]:
            pills.append(ev)
    if pills:
        pills_html = "".join(f'<div class="pill">{p}</div>' for p in pills)
        st.markdown(f'<div class="pill-row">{pills_html}</div>', unsafe_allow_html=True)
    if conv_dim and conv_dim.get("summary"):
        st.markdown(f'<div class="insight-box"><strong>Convergence:</strong> {conv_dim["summary"]}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Alert Panel
# ---------------------------------------------------------------------------

def _render_alert_panel(report: dict, df: Optional[pd.DataFrame]) -> None:
    st.markdown('<div class="section-title">Alert Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Detected anomalies and stagnation periods requiring attention</div>',
                unsafe_allow_html=True)

    periods = report.get("stagnation_periods", [])
    if not periods:
        st.success("No stagnation periods detected.")
        return

    all_severities = ["critical", "high", "warning"]
    present_severities = sorted(
        {p.get("severity", "warning") for p in periods},
        key=lambda s: all_severities.index(s) if s in all_severities else len(all_severities),
    )
    if len(present_severities) > 1:
        selected = st.multiselect(
            "Filter by severity",
            options=present_severities,
            default=present_severities,
            format_func=lambda s: s.capitalize(),
        )
        periods = [p for p in periods if p.get("severity", "warning") in selected]
        if not periods:
            st.info("No alerts match the selected severity filters.")
            return

    def _alert_group_key(p: dict) -> tuple:
        llm = p.get("llm_analysis")
        llm_key = (llm.get("category", ""), llm.get("recommendation", "")) if isinstance(llm, dict) else llm
        return (p.get("severity", "warning"), p.get("dominant_failure_type", ""), p.get("recommendation", ""), llm_key)

    from collections import defaultdict
    groups: dict = defaultdict(list)
    for p in periods:
        groups[_alert_group_key(p)].append(p)

    for key, group in groups.items():
        severity = key[0]
        dominant = key[1]
        recommendation = key[2]
        first = group[0]
        llm_analysis = first.get("llm_analysis")
        all_crash_samples = [s for p in group for s in (p.get("crash_samples") or [])]

        if len(group) == 1:
            start = first.get("start_iteration", "?")
            end = first.get("end_iteration") if first.get("end_iteration") is not None else "ongoing"
            length = first.get("length", "?")
            title = f"{length} consecutive non-improving iterations (iters {start}–{end})"
        else:
            ranges = [
                f"iters {p.get('start_iteration','?')}–{p.get('end_iteration', 'ongoing')}"
                for p in group
            ]
            if len(ranges) <= 4:
                ranges_str = ", ".join(ranges)
            else:
                ranges_str = f"{ranges[0]}, {ranges[1]}, … {ranges[-1]} (+{len(ranges)-2} more)"
            title = f"{len(group)} occurrences ({ranges_str})"

        root_html = (f'<div><span class="root-cause-badge">{dominant.upper()}</span></div>'
                     if dominant else "")

        crash_html = ""
        if all_crash_samples:
            items = "".join(
                f'<li><strong>iter {s["iteration"]}:</strong> <code style="color:#d93025">{s["error"]}</code></li>'
                for s in all_crash_samples
            )
            crash_html = f'<div class="llm-block"><strong>Crash errors:</strong><ul>{items}</ul></div>'

        llm_html = ""
        if llm_analysis:
            if isinstance(llm_analysis, dict):
                category = llm_analysis.get("category", "")
                llm_rec = llm_analysis.get("recommendation", "")
                parts = []
                if category:
                    parts.append(f"<strong>Category:</strong> {category}")
                if llm_rec:
                    parts.append(f"<strong>LLM insight:</strong> {llm_rec}")
                if parts:
                    llm_html = f'<div class="llm-block">{" &nbsp;·&nbsp; ".join(parts)}</div>'
            else:
                llm_html = f'<div class="llm-block">{llm_analysis}</div>'

        rec_html = (f'<div class="rec-box"><strong>Recommendation:</strong> {recommendation}</div>'
                    if recommendation else "")

        iters_html = ""
        if len(group) > 1:
            iter_items = "".join(
                f'<li>iters {p.get("start_iteration","?")}–{p.get("end_iteration","?")}: {p.get("length","?")} consecutive non-improving</li>'
                for p in group
            )
            iters_html = f'<div class="llm-block"><strong>Occurrences:</strong><ul>{iter_items}</ul></div>'

        badge_html = _badge(severity)
        with st.expander(f"{severity.upper()} — {title}", expanded=(severity == "critical")):
            st.markdown(f"""
            {badge_html} <span style="font-size:13px;font-weight:600">{title}</span>
            {root_html}
            {iters_html}
            {crash_html}
            {llm_html}
            {rec_html}
            """, unsafe_allow_html=True)

    novel = report.get("novel_observations", [])
    if novel:
        items = "".join(f"<li>{obs}</li>" for obs in novel)
        st.markdown(f"""
        <div class="obs-section">
          <h4>Novel Observations</h4>
          <ul>{items}</ul>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Failure Budget
# ---------------------------------------------------------------------------

FAILURE_COLORS = {
    "success": "#34a853", "partial": "#81c995", "worse": "#e37400",
    "wrong_output": "#d93025", "timeout": "#9aa0a6", "crash": "#7b2218",
    "format_invalid": "#f9ab00",
}


def _render_failure_budget(report: dict, df: Optional[pd.DataFrame]) -> None:
    agg = report.get("aggregate_stats", {})
    total = agg.get("total_iterations", len(df) if df is not None else 0)
    st.markdown('<div class="section-title">Failure Budget</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Distribution of outcomes across {total} iterations</div>',
                unsafe_allow_html=True)

    if df is None:
        st.info("No iteration DataFrame available.")
        return

    failure_col = next((c for c in ["failure_mode", "evaluation_status"] if c in df.columns), None)
    if not failure_col:
        st.info("No failure_mode or evaluation_status column found.")
        return

    counts = df[failure_col].value_counts()
    n = len(df)

    segs = []
    legend_items = []
    for mode, cnt in counts.items():
        pct = cnt / n * 100
        color = FAILURE_COLORS.get(str(mode), "#9aa0a6")
        txt_color = "#202124" if str(mode) == "format_invalid" else "#fff"
        if pct >= 5:
            segs.append(f'<div class="seg" style="width:{pct:.1f}%;background:{color};color:{txt_color}">{pct:.0f}%</div>')
        else:
            segs.append(f'<div class="seg" style="width:{pct:.1f}%;background:{color}"></div>')
        legend_items.append(
            f'<div class="budget-legend-item"><div class="color-sq" style="background:{color}"></div>{mode} ({pct:.1f}%)</div>'
        )

    st.markdown(f"""
    <div class="budget-bar-wrap">
      <div class="stacked-bar">{"".join(segs)}</div>
      <div class="budget-legend">{"".join(legend_items)}</div>
    </div>
    """, unsafe_allow_html=True)

    rows = ""
    for mode, cnt in counts.items():
        pct = cnt / n * 100
        color = FAILURE_COLORS.get(str(mode), "#9aa0a6")
        avg_delta = "—"
        if "score_delta" in df.columns:
            subset = df[df[failure_col] == mode]["score_delta"].dropna()
            if not subset.empty:
                avg_d = subset.mean()
                sign = "+" if avg_d > 0 else ""
                avg_delta = f"{sign}{avg_d:.3f}"
        rows += f"""<tr>
          <td><span style="color:{color};font-weight:600">{mode}</span></td>
          <td>{cnt}</td><td>{pct:.1f}%</td><td>{avg_delta}</td>
        </tr>"""

    st.markdown(f"""
    <table class="data-table">
      <thead><tr><th>Mode</th><th>Count</th><th>%</th><th>Avg Δscore</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    if "cascade_stage_failed" in df.columns:
        import plotly.express as px
        csf = df["cascade_stage_failed"].dropna().value_counts().sort_index()
        if not csf.empty:
            st.markdown('<div class="section-title" style="margin-top:20px">Cascade Stage Failures</div>',
                        unsafe_allow_html=True)
            fig = px.bar(x=csf.index.astype(str), y=csf.values,
                         labels={"x": "Stage", "y": "Count"},
                         color_discrete_sequence=["#e37400"])
            fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff",
                               height=220, margin=dict(l=40, r=20, t=20, b=40))
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel: Mutation Effectiveness
# ---------------------------------------------------------------------------

def _render_mutation_effectiveness(report: dict, df: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go

    st.markdown('<div class="section-title">Mutation Effectiveness</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Success rate breakdown by mutation type, model, and island</div>',
                unsafe_allow_html=True)

    if df is None or "mutation_type" not in df.columns:
        st.info("No mutation type data available.")
        return

    work_df = df.copy()
    if "score_delta" not in work_df.columns:
        if "child_score" in work_df.columns and "parent_score" in work_df.columns:
            work_df["score_delta"] = work_df["child_score"] - work_df["parent_score"]

    has_delta = "score_delta" in work_df.columns
    if has_delta:
        work_df["improved"] = work_df["score_delta"] > 0.001

    mut_gb = work_df.groupby("mutation_type")
    mut_summary = mut_gb.agg(attempts=("mutation_type", "count")).reset_index()
    if has_delta:
        mut_summary = mut_summary.merge(
            mut_gb.agg(successes=("improved", "sum"), mean_delta=("score_delta", "mean")).reset_index(),
            on="mutation_type"
        )
        mut_summary["success_pct"] = mut_summary["successes"] / mut_summary["attempts"].clip(lower=1) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="panel-card"><h4>By Mutation Type</h4>', unsafe_allow_html=True)
        if has_delta and "success_pct" in mut_summary.columns:
            fig = go.Figure(go.Bar(
                x=mut_summary["mutation_type"], y=mut_summary["success_pct"],
                marker_color="#1a73e8",
                text=mut_summary["success_pct"].map(lambda x: f"{x:.0f}%"),
                textposition="outside",
            ))
            fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff", height=220,
                               margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                               yaxis=dict(range=[0, 115], title="Success %"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(mut_summary)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel-card"><h4>By Model</h4>', unsafe_allow_html=True)
        if "model" in work_df.columns and has_delta:
            model_gb = work_df.groupby("model")
            model_sum = model_gb.agg(
                attempts=("model", "count"), successes=("improved", "sum")
            ).reset_index()
            model_sum["rate"] = model_sum["successes"] / model_sum["attempts"].clip(lower=1) * 100
            for _, row in model_sum.iterrows():
                rate = row["rate"]
                color = "#34a853" if rate >= 55 else ("#e37400" if rate >= 40 else "#d93025")
                st.markdown(f"""
                <div style="margin-bottom:14px">
                  <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
                    <span style="font-weight:600">{row["model"]}</span>
                    <span style="color:{color};font-weight:700">{rate:.0f}%</span>
                  </div>
                  <div style="background:#e0e0e0;border-radius:4px;height:12px;overflow:hidden">
                    <div style="width:{min(rate,100):.0f}%;height:100%;background:{color};border-radius:4px"></div>
                  </div>
                  <div style="font-size:10px;color:#9aa0a6;margin-top:2px">{int(row["attempts"])} iters</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No model column found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="panel-card"><h4>By Island</h4>', unsafe_allow_html=True)
        if "island" in work_df.columns and has_delta:
            isl_gb = work_df.groupby("island")
            isl_sum = isl_gb.agg(
                attempts=("island", "count"), successes=("improved", "sum")
            ).reset_index()
            isl_sum["rate"] = isl_sum["successes"] / isl_sum["attempts"].clip(lower=1) * 100
            rows = ""
            for _, row in isl_sum.iterrows():
                rate = row["rate"]
                color = "#34a853" if rate >= 60 else ("#e37400" if rate >= 45 else "#d93025")
                rows += f"""<tr>
                  <td style="padding:6px">Island {row["island"]}</td>
                  <td style="padding:6px;text-align:center">{int(row["attempts"])}</td>
                  <td style="padding:6px;text-align:center;color:{color};font-weight:700">{rate:.0f}%</td>
                </tr>"""
            st.markdown(f"""
            <table style="width:100%;font-size:12px;border-collapse:collapse">
              <thead><tr>
                <th style="text-align:left;padding:5px 6px;color:#5f6368;border-bottom:1px solid #e0e0e0">Island</th>
                <th style="padding:5px 6px;color:#5f6368;border-bottom:1px solid #e0e0e0;text-align:center">Iters</th>
                <th style="padding:5px 6px;color:#5f6368;border-bottom:1px solid #e0e0e0;text-align:center">Success</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.info("No island column found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Cross table
    if "island" in work_df.columns and has_delta:
        cross = (pd.crosstab(work_df["mutation_type"], work_df["island"],
                             values=work_df["improved"], aggfunc="mean") * 100)
        islands = cross.columns.tolist()
        hdrs = "".join(f"<th>Island {i}</th>" for i in islands) + "<th>Overall</th>"
        rows_html = ""
        for mt in cross.index:
            cells = f'<td class="row-head">{mt}</td>'
            for isl in islands:
                val = cross.loc[mt, isl] if isl in cross.columns else float("nan")
                if pd.isna(val):
                    cells += '<td class="heat-neutral">—</td>'
                else:
                    cells += f'<td class="{_heat(val)}" style="font-weight:700">{val:.0f}%</td>'
            ov = work_df[work_df["mutation_type"] == mt]["improved"].mean() * 100
            cells += f'<td class="{_heat(ov)}" style="font-weight:700">{ov:.0f}%</td>'
            rows_html += f"<tr>{cells}</tr>"
        st.markdown(f"""
        <div class="panel-card" style="margin-top:16px">
          <h4>Mutation Type × Island — Success Rate</h4>
          <table class="cross-table">
            <thead><tr><th></th>{hdrs}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Compliance × Score
# ---------------------------------------------------------------------------

def _render_compliance_score(report: dict, df: Optional[pd.DataFrame]) -> None:
    st.markdown('<div class="section-title">Compliance × Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Relationship between instruction compliance and outcomes</div>',
                unsafe_allow_html=True)

    if df is None:
        st.info("No iteration DataFrame available.")
        return

    comp_col = next((c for c in ["compliance_level", "compliance_status", "format_valid"] if c in df.columns), None)
    if not comp_col:
        st.info("No compliance column found.")
        return

    work_df = df.copy()
    if "score_delta" not in work_df.columns:
        if "child_score" in work_df.columns and "parent_score" in work_df.columns:
            work_df["score_delta"] = work_df["child_score"] - work_df["parent_score"]

    rows = ""
    for level, grp in work_df.groupby(comp_col):
        cnt = len(grp)
        success_rate = "—"
        avg_delta = "—"
        avg_score = "—"
        hc = ""

        if "score_delta" in work_df.columns:
            improved = (grp["score_delta"] > 0.001).sum()
            rate = improved / cnt * 100 if cnt > 0 else 0
            success_rate = f"{rate:.0f}%"
            hc = _heat(rate)
            avg_d = grp["score_delta"].dropna().mean()
            if not pd.isna(avg_d):
                avg_delta = f"{'+' if avg_d > 0 else ''}{avg_d:.3f}"

        if "child_score" in work_df.columns:
            avg_s = grp["child_score"].dropna().mean()
            if not pd.isna(avg_s):
                avg_score = f"{avg_s:.3f}"

        level_str = str(level)
        if "non" in level_str.lower():
            badge_sty = 'style="background:#fce8e6;border-radius:4px;padding:2px 8px;font-size:12px;color:#c5221f;font-weight:700"'
        elif "partial" in level_str.lower():
            badge_sty = 'style="background:#fef7e0;border-radius:4px;padding:2px 8px;font-size:12px;color:#8a6400;font-weight:700"'
        else:
            badge_sty = 'style="background:#e8f0fe;border-radius:4px;padding:2px 8px;font-size:12px;color:#1a55c0;font-weight:700"'

        rows += f"""<tr>
          <td><strong>{level}</strong></td>
          <td><span {badge_sty}>{cnt} iters</span></td>
          <td class="{hc}" style="font-weight:700;font-size:13px">{success_rate}</td>
          <td class="{hc}" style="font-weight:700;font-size:13px">{avg_delta}</td>
          <td class="{hc}" style="font-weight:700;font-size:13px">{avg_score}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:20px;margin-bottom:20px">
      <table class="dim-table">
        <thead><tr>
          <th>Compliance Level</th><th>Count</th><th>Success Rate</th>
          <th>Avg Δscore</th><th>Avg Abs Score</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    comp_dim = next((d for d in report.get("dimensions", []) if "compliance" in d.get("name", "").lower()), None)
    if comp_dim and comp_dim.get("summary"):
        st.markdown(f'<div class="insight-box"><strong>Key Insight:</strong> {comp_dim["summary"]}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Cascade Stages
# ---------------------------------------------------------------------------

def _render_cascade_stages(report: dict, df: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go

    st.markdown('<div class="section-title">Cascade Stage Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Where in the evaluation pipeline do failures occur?</div>',
                unsafe_allow_html=True)

    if df is None or "cascade_stage_failed" not in df.columns:
        st.info("No cascade_stage_failed column found in the DataFrame.")
        return

    csf = df["cascade_stage_failed"].value_counts().sort_index()
    passed = int((df["cascade_stage_failed"].isna()).sum())

    stages = list(csf.index.astype(str)) + ["Passed All"]
    counts = list(csf.values) + [passed]
    palette = ["#d93025", "#e37400", "#f9ab00", "#fbbc04", "#34a853"]
    colors = [palette[min(i, len(palette) - 2)] for i in range(len(stages) - 1)] + ["#34a853"]

    fig = go.Figure(go.Bar(x=stages, y=counts, marker_color=colors,
                            text=counts, textposition="outside"))
    fig.update_layout(xaxis_title="Evaluation Stage", yaxis_title="Count",
                      plot_bgcolor="#fff", paper_bgcolor="#fff",
                      height=300, margin=dict(l=40, r=20, t=30, b=40))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f3f4")
    st.plotly_chart(fig, use_container_width=True)

    eval_dim = next((d for d in report.get("dimensions", []) if "evaluator" in d.get("name", "").lower()), None)
    if eval_dim and eval_dim.get("summary"):
        st.markdown(f'<div class="insight-box"><strong>Evaluator Output:</strong> {eval_dim["summary"]}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Evaluator Artifacts
# ---------------------------------------------------------------------------

def _render_artifact_clusters(report: dict) -> None:
    st.markdown('<div class="section-title">Evaluator Artifact Clusters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Recurring error patterns detected across evaluation runs</div>',
                unsafe_allow_html=True)

    clusters = report.get("artifact_clusters")

    if not clusters:
        stag_dim = next((d for d in report.get("dimensions", []) if d.get("name") == "Stagnation"), None)
        if stag_dim:
            for ev in stag_dim.get("evidence", []):
                st.markdown(f'<div class="artifact-card"><div style="padding:4px 0;font-size:13px">{ev}</div></div>',
                            unsafe_allow_html=True)
        else:
            st.info("No artifact cluster data available.")
    elif isinstance(clusters, list):
        total_occ = sum(c.get("count", 1) for c in clusters if isinstance(c, dict))
        for cluster in clusters:
            if isinstance(cluster, dict):
                name = cluster.get("name", cluster.get("label", "Unknown"))
                count = cluster.get("count", cluster.get("occurrences", 1))
                root_cause = cluster.get("root_cause", cluster.get("description", ""))
                rec = cluster.get("recommendation", "")
                pct = count / max(total_occ, 1) * 100
                bar_color = "#1a73e8" if pct > 50 else ("#e37400" if pct > 20 else "#9aa0a6")
                root_html = f'<div style="font-size:12px;color:#5f6368;margin-bottom:8px"><strong>Root cause:</strong> {root_cause}</div>' if root_cause else ""
                rec_html = f'<div class="rec-box"><strong>Recommendation:</strong> {rec}</div>' if rec else ""
                st.markdown(f"""
                <div class="artifact-card">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
                    <div style="flex:1">
                      <div style="font-size:13px;font-weight:700;color:#202124;margin-bottom:6px">
                        <code style="background:#f1f3f4;padding:2px 6px;border-radius:3px;font-size:12px">{name}</code>
                      </div>
                      <div class="artifact-freq-bar" style="width:{min(pct,100):.1f}%;background:{bar_color}"></div>
                      <div class="freq-label">{pct:.1f}% ({count} occurrences)</div>
                    </div>
                    <span class="count-badge">{count}</span>
                  </div>
                  {root_html}{rec_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="artifact-card"><div style="padding:4px 0">{cluster}</div></div>',
                            unsafe_allow_html=True)
    elif isinstance(clusters, dict):
        for k, v in clusters.items():
            st.markdown(f"""
            <div class="artifact-card">
              <div style="font-size:13px;font-weight:700;margin-bottom:6px">{k}</div>
              <div style="font-size:12px;color:#5f6368">{v}</div>
            </div>
            """, unsafe_allow_html=True)

    cross_dim = report.get("cross_dimension_interactions", "")
    if cross_dim:
        st.markdown(f'<div class="insight-box"><strong>Cross-dimension interactions:</strong> {cross_dim}</div>',
                    unsafe_allow_html=True)

    novel = report.get("novel_observations", [])
    if novel:
        items = "".join(f"<li>{obs}</li>" for obs in novel)
        st.markdown(f'<div class="obs-section"><h4>Novel Observations</h4><ul>{items}</ul></div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Efficiency Curve
# ---------------------------------------------------------------------------

def _render_efficiency_curve(report: dict, df: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go

    st.markdown('<div class="section-title">Efficiency Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cumulative improvement vs. iterations — where was compute well spent?</div>',
                unsafe_allow_html=True)

    eff_dim = next((d for d in report.get("dimensions", []) if d.get("name") == "Efficiency"), None)

    if df is not None and "child_score" in df.columns:
        work_df = df.copy()
        if "iteration" not in work_df.columns:
            work_df["iteration"] = range(len(work_df))

        best_curve = work_df["child_score"].expanding().max()
        initial = best_curve.iloc[0] if len(best_curve) else 0.0
        cumulative = best_curve - initial

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for p in report.get("stagnation_periods", []):
                x0, x1 = p.get("start_iteration", 0), p.get("end_iteration") or work_df["iteration"].max()
                sev = p.get("severity", "")
                fill = "rgba(217,48,37,0.08)" if sev == "critical" else "rgba(249,171,0,0.10)"
                fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_color="rgba(0,0,0,0)")
            fig.add_trace(go.Scatter(x=work_df["iteration"], y=cumulative, mode="lines",
                                     name="Cumulative Improvement",
                                     line=dict(color="#1a73e8", width=2),
                                     fill="tozeroy", fillcolor="rgba(26,115,232,0.05)"))
            fig.update_layout(title="Cumulative Improvement", xaxis_title="Iteration",
                               yaxis_title="Δscore from start", plot_bgcolor="#fff",
                               paper_bgcolor="#fff", height=280,
                               margin=dict(l=40, r=20, t=40, b=40))
            fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
            fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=work_df["iteration"], y=best_curve, mode="lines",
                                      name="Best Score",
                                      line=dict(color="#34a853", width=2, dash="dash")))
            fig2.update_layout(title="Best Score vs. Iteration", xaxis_title="Iteration",
                                yaxis_title="Best Score", plot_bgcolor="#fff",
                                paper_bgcolor="#fff", height=280,
                                margin=dict(l=40, r=20, t=40, b=40))
            fig2.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
            fig2.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No iteration DataFrame available for efficiency curve.")

    # Metric cards from efficiency dimension evidence
    if eff_dim:
        evs = eff_dim.get("evidence", [])[:4]
        if evs:
            cards = "".join(
                f'<div class="metric-card"><div class="m-label">Evidence {i+1}</div>'
                f'<div class="m-val" style="font-size:11px;line-height:1.4">{ev[:80]}</div></div>'
                for i, ev in enumerate(evs)
            )
            st.markdown(f'<div class="metric-4">{cards}</div>', unsafe_allow_html=True)

        if eff_dim.get("summary"):
            st.markdown(f'<div class="insight-box"><strong>Efficiency:</strong> {eff_dim["summary"]}</div>',
                        unsafe_allow_html=True)
        rec = eff_dim.get("recommendation", "")
        if rec:
            if (eff_dim.get("rating") or 5) <= 2:
                st.warning(f"Recommendation: {rec}")
            else:
                st.success(f"Recommendation: {rec}")


# ---------------------------------------------------------------------------
# Panel: Explore vs Exploit
# ---------------------------------------------------------------------------

def _render_exploration(report: dict, df: Optional[pd.DataFrame]) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<div class="section-title">Exploration vs. Exploitation Timeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Phase classification, mutation strategy mix, and per-mutation score impact</div>',
                unsafe_allow_html=True)

    expl_dim = next((d for d in report.get("dimensions", []) if d.get("name") == "Exploration"), None)

    if df is not None and "mutation_type" in df.columns:
        work_df = df.copy()
        if "iteration" not in work_df.columns:
            work_df["iteration"] = range(len(work_df))

        top_muts = work_df["mutation_type"].value_counts().head(8).index.tolist()
        work_df["mutation_type_grouped"] = work_df["mutation_type"].apply(
            lambda x: x if x in top_muts else "other"
        )

        col1, col2 = st.columns(2)

        with col1:
            dummies = pd.get_dummies(work_df.set_index("iteration")["mutation_type_grouped"])
            window = max(1, len(dummies) // 15)
            smoothed = dummies.rolling(window=window, min_periods=1).mean().reset_index()

            fig = px.area(smoothed, x="iteration", y=[c for c in smoothed.columns if c != "iteration"],
                          labels={"value": "Proportion", "variable": "Mutation Type"},
                          title="Mutation Type Mix Over Time")
            fig.update_layout(plot_bgcolor="#fff", paper_bgcolor="#fff", height=320,
                               margin=dict(l=40, r=20, t=40, b=70),
                               legend=dict(orientation="h", y=-0.45))
            fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "score_delta" in work_df.columns:
                delta_df = work_df.dropna(subset=["score_delta"])
                mut_order = (
                    delta_df.groupby("mutation_type_grouped")["score_delta"]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                fig2 = go.Figure()
                for mt in mut_order:
                    vals = delta_df.loc[delta_df["mutation_type_grouped"] == mt, "score_delta"]
                    fig2.add_trace(go.Box(y=vals, name=mt, boxmean=True, marker_size=3))
                fig2.add_hline(y=0, line_dash="dash", line_color="#9aa0a6", line_width=1)
                fig2.update_layout(
                    title="Score Δ Distribution by Mutation Type",
                    yaxis_title="Score Δ",
                    plot_bgcolor="#fff", paper_bgcolor="#fff", height=320,
                    margin=dict(l=40, r=20, t=40, b=70),
                    showlegend=False,
                )
                fig2.update_xaxes(showgrid=False)
                fig2.update_yaxes(showgrid=True, gridcolor="#e0e0e0", zeroline=False)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No `score_delta` column available for distribution chart.")

        if "score_delta" in work_df.columns:
            success_df = work_df.copy()
            success_df["success"] = success_df["score_delta"].fillna(0) > 0
            sr = (
                success_df.groupby("mutation_type_grouped")["success"]
                .agg(success_rate="mean", count="size")
                .reset_index()
                .sort_values("success_rate", ascending=False)
            )
            fig3 = px.bar(
                sr, x="mutation_type_grouped", y="success_rate",
                text=sr["count"].apply(lambda n: f"n={n}"),
                labels={"mutation_type_grouped": "Mutation Type", "success_rate": "Success Rate"},
                title="Positive-Δ Success Rate by Mutation Type",
                color="success_rate",
                color_continuous_scale=["#ea4335", "#fbbc04", "#34a853"],
                range_color=[0, 1],
            )
            fig3.update_traces(textposition="outside")
            fig3.update_layout(
                plot_bgcolor="#fff", paper_bgcolor="#fff", height=280,
                margin=dict(l=40, r=20, t=40, b=60),
                coloraxis_showscale=False,
            )
            fig3.update_yaxes(tickformat=".0%", range=[0, 1.15], showgrid=True, gridcolor="#e0e0e0")
            fig3.update_xaxes(showgrid=False)
            st.plotly_chart(fig3, use_container_width=True)

        total = len(work_df)
        pills = [
            f'<div class="pill"><strong>{t}:</strong> {cnt} ({cnt/total*100:.0f}%)</div>'
            for t, cnt in work_df["mutation_type_grouped"].value_counts().items()
        ][:6]
        st.markdown(f'<div class="pill-row">{"".join(pills)}</div>', unsafe_allow_html=True)

    elif df is None:
        st.info("No iteration DataFrame available.")

    # ── Evidence from report ───────────────────────────────────────────────────
    if expl_dim:
        evs = expl_dim.get("evidence", [])

        _QUANT_PREFIXES = (
            "Structural diversity index",
            "Exploit phase fraction",
            "Explore phase fraction",
            "Distinct strategy clusters",
            "Revert frequency",
        )

        quant_items = [e for e in evs if any(e.startswith(p) for p in _QUANT_PREFIXES)]
        qual_items  = [e for e in evs if not any(e.startswith(p) for p in _QUANT_PREFIXES)]

        # Quantitative summary: compact bullet list
        if quant_items:
            bullets_html = "".join(
                f'<li style="margin-bottom:3px">{item}</li>' for item in quant_items
            )
            st.markdown(
                f'<ul style="font-size:13px;color:#3c4043;margin:12px 0 16px 16px;padding:0">'
                f'{bullets_html}</ul>',
                unsafe_allow_html=True,
            )

        # Qualitative LLM-judge evidence: group headers + indented children
        # Non-indented lines are section headers; "  • " lines are violations;
        # "    Fix:" lines are fix suggestions for the last violation.
        judge_groups: list = []
        current: dict | None = None
        for ev in qual_items:
            if ev.startswith("    Fix:"):
                if current and current["children"]:
                    current["children"][-1]["fix"] = ev[len("    Fix:"):].strip()
            elif ev.startswith("  • "):
                if current is not None:
                    current["children"].append({"text": ev[4:], "fix": None})
            else:
                if current is not None:
                    judge_groups.append(current)
                current = {"header": ev, "children": []}
        if current is not None:
            judge_groups.append(current)

        for group in judge_groups:
            header = group["header"]
            children = group["children"]

            violations_html = ""
            for child in children:
                fix_html = ""
                if child["fix"]:
                    fix_html = (
                        f'<div style="margin-top:6px;padding:6px 10px;background:#e8f5e9;'
                        f'border-left:3px solid #34a853;border-radius:3px;font-size:11px;color:#1e7e34">'
                        f'<strong>Fix:</strong> {child["fix"]}</div>'
                    )
                violations_html += (
                    f'<div style="margin-top:8px;padding:8px 10px;background:#fafafa;'
                    f'border:1px solid #e0e0e0;border-radius:4px;font-size:12px;color:#3c4043">'
                    f'{child["text"]}{fix_html}</div>'
                )

            st.markdown(
                f'<div style="background:#fff;border:1px solid #dadce0;border-radius:8px;'
                f'padding:14px 16px;margin-bottom:10px">'
                f'<div style="font-size:13px;font-weight:600;color:#202124;margin-bottom:{"8px" if violations_html else "0"}">'
                f'{header}</div>'
                f'{violations_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div class="insight-box"><strong>Exploration Analysis:</strong> {expl_dim.get("summary","")}</div>',
                    unsafe_allow_html=True)
        rec = expl_dim.get("recommendation", "")
        if rec:
            if (expl_dim.get("rating") or 5) <= 2:
                st.warning(f"Recommendation: {rec}")
            else:
                st.success(f"Recommendation: {rec}")


# ---------------------------------------------------------------------------
# Panel: Search Space
# ---------------------------------------------------------------------------

def _render_search_space(report: dict, df: Optional[pd.DataFrame]) -> None:
    import re
    from collections import Counter
    import plotly.graph_objects as go

    st.markdown('<div class="section-title">Search Space Coverage</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Parameter diversity, frozen dimensions, and trial coverage</div>',
        unsafe_allow_html=True,
    )

    ss_dim = next((d for d in report.get("dimensions", []) if d.get("name") == "Search Space"), None)
    if not ss_dim:
        st.info("No Search Space dimension data available.")
        return

    rating = ss_dim.get("rating")
    summary = ss_dim.get("summary", "")
    evs = ss_dim.get("evidence", [])

    st.markdown(
        f'<div class="insight-box">{_rating_badge(rating)} <strong>Search Space:</strong> {summary}</div>',
        unsafe_allow_html=True,
    )

    # ── Parse structured values from evidence strings ─────────────────────────
    eff_dim: Optional[int] = None
    trial_ratio: Optional[float] = None
    frozen_params: list = []
    bound_hit: list = []

    for ev in evs:
        if ev.startswith("Effective dimensionality:"):
            try:
                eff_dim = int(ev.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif ev.startswith("Trial-to-param ratio:"):
            try:
                trial_ratio = float(ev.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif "frozen" in ev.lower() and "[" in ev:
            m = re.search(r"\[([^\]]+)\]", ev)
            if m:
                frozen_params = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        elif ev.startswith("Bound-hit params:") and "none" not in ev.lower():
            try:
                bound_hit = eval(ev.split(":", 1)[1].strip())  # noqa: S307
            except (ValueError, SyntaxError):
                pass

    # ── Key metrics ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Effective Dimensions", eff_dim if eff_dim is not None else "—")
    with col2:
        st.metric("Trial-to-Param Ratio", f"{trial_ratio:.1f}" if trial_ratio is not None else "—")
    with col3:
        n_frozen = len(frozen_params)
        st.metric(
            "Frozen Parameters",
            n_frozen,
            delta=f"−{n_frozen} diversity" if n_frozen else None,
            delta_color="inverse",
        )

    # ── Parameter distribution charts ─────────────────────────────────────────
    if df is not None and "parameters" in df.columns:
        param_series = df["parameters"].dropna()
        dicts = [d for d in param_series if isinstance(d, dict) and d]
        if dicts:
            param_keys = sorted({k for d in dicts for k in d})
            st.markdown(
                '<div class="section-title" style="font-size:14px;margin-top:20px">Parameter Distributions</div>',
                unsafe_allow_html=True,
            )
            for key in param_keys:
                vals = [d[key] for d in dicts if key in d]
                is_frozen = key in frozen_params
                is_bound = key in bound_hit

                frozen_badge = (
                    ' <span style="background:#fce8e6;color:#c5221f;font-size:10px;'
                    'padding:2px 7px;border-radius:10px;font-weight:600">FROZEN</span>'
                    if is_frozen else ""
                )
                bound_badge = (
                    ' <span style="background:#fff3e0;color:#e37400;font-size:10px;'
                    'padding:2px 7px;border-radius:10px;font-weight:600">HITS BOUND</span>'
                    if is_bound else ""
                )
                st.markdown(
                    f'<div style="font-size:13px;font-weight:600;color:#202124;margin:14px 0 4px">'
                    f'{key}{frozen_badge}{bound_badge}</div>',
                    unsafe_allow_html=True,
                )

                numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                is_numeric = len(numeric_vals) == len(vals)

                bar_color = "#ea4335" if is_frozen else "#1a73e8"

                if is_numeric:
                    if len(set(numeric_vals)) == 1:
                        st.markdown(
                            f'<div style="font-size:12px;color:#5f6368;padding:6px 0">'
                            f'Always <strong>{numeric_vals[0]}</strong> — no variance across {len(vals)} iterations.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        fig = go.Figure(go.Histogram(
                            x=numeric_vals,
                            nbinsx=min(20, len(set(numeric_vals))),
                            marker_color=bar_color,
                        ))
                        fig.update_layout(
                            plot_bgcolor="#fff", paper_bgcolor="#fff",
                            height=160, margin=dict(l=30, r=10, t=10, b=30),
                            showlegend=False,
                        )
                        fig.update_xaxes(showgrid=False)
                        fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", title_text="count")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    counts = Counter(str(v) for v in vals)
                    labels = list(counts.keys())
                    values_cnt = list(counts.values())
                    fig = go.Figure(go.Bar(x=labels, y=values_cnt, marker_color=bar_color))
                    fig.update_layout(
                        plot_bgcolor="#fff", paper_bgcolor="#fff",
                        height=160, margin=dict(l=30, r=10, t=10, b=30),
                        showlegend=False,
                    )
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", title_text="count")
                    st.plotly_chart(fig, use_container_width=True)
    elif df is None:
        st.info("No iteration DataFrame available for parameter charts.")

    # ── Full evidence expander ────────────────────────────────────────────────
    if evs:
        with st.expander("Full evidence"):
            for ev in evs:
                st.markdown(f"- {ev}")

    # ── Recommendation ────────────────────────────────────────────────────────
    rec = ss_dim.get("recommendation", "")
    if rec:
        if rating is not None and rating <= 2:
            st.warning(f"**Recommendation:** {rec}")
        else:
            st.success(f"**Recommendation:** {rec}")


# ---------------------------------------------------------------------------
# Panel: Meta-Analysis
# ---------------------------------------------------------------------------

def _render_meta_analysis(report: dict, df: Optional[pd.DataFrame]) -> None:
    st.markdown('<div class="section-title">Meta-Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">LLM self-guidance quality: suggestion adherence, reasoning traces, and scratchpad dynamics</div>',
        unsafe_allow_html=True,
    )

    meta_dim = next((d for d in report.get("dimensions", []) if d.get("name") == "Meta-Analysis"), None)
    if not meta_dim:
        st.info("No Meta-Analysis dimension data found in report.")
        return

    rating = meta_dim.get("rating")
    summary = meta_dim.get("summary", "")
    evidence = meta_dim.get("evidence", [])
    recommendation = meta_dim.get("recommendation", "")

    # ── Status box ────────────────────────────────────────────────────────────
    if rating is None:
        status_html = (
            '<div style="background:#f1f3f4;border-left:4px solid #9aa0a6;border-radius:0 6px 6px 0;'
            'padding:12px 16px;font-size:13px;color:#5f6368;line-height:1.6;margin-bottom:16px">'
            '<strong style="color:#5f6368">Not Applicable</strong> — '
            f'{summary}</div>'
        )
    else:
        status_html = (
            f'<div class="insight-box"><strong>Meta-Analysis ({rating}/5):</strong> {summary}</div>'
        )
    st.markdown(status_html, unsafe_allow_html=True)

    # ── Metric cards for quantitative fields ──────────────────────────────────
    # Parse key values from evidence strings
    def _extract(evs, key):
        for e in evs:
            if key.lower() in e.lower():
                return e
        return None

    sfr = _extract(evidence, "suggestion follow rate")
    cir = _extract(evidence, "conditional improvement rate")
    prf = _extract(evidence, "pattern reuse frequency")
    sgr = _extract(evidence, "scratchpad growth rate")
    cmp = _extract(evidence, "compaction events")

    quant_cards = [c for c in [sfr, cir, prf, sgr, cmp] if c]
    if quant_cards:
        cards_html = "".join(
            f'<div class="metric-card"><div class="m-label">{c.split(":")[0].strip()}</div>'
            f'<div class="m-val" style="font-size:13px;font-weight:600">'
            f'{c.split(":", 1)[1].strip() if ":" in c else "—"}</div></div>'
            for c in quant_cards
        )
        st.markdown(f'<div class="metric-4">{cards_html}</div>', unsafe_allow_html=True)

    # ── Full evidence list ────────────────────────────────────────────────────
    other_evidence = [e for e in evidence if e not in quant_cards]
    if other_evidence:
        st.markdown('<div class="section-title" style="margin-top:20px">Evidence</div>', unsafe_allow_html=True)
        for ev in other_evidence:
            st.markdown(f"- {ev}")

    # ── What to log ───────────────────────────────────────────────────────────
    if rating is None:
        st.markdown('<div class="section-title" style="margin-top:20px">Required Fields</div>', unsafe_allow_html=True)
        st.markdown("""
The following fields must be present in iteration records to enable meta-analysis:

| Field | Description |
|---|---|
| `meta_suggestion` | The suggestion the LLM proposed for the next iteration |
| `followed_suggestion` | `true` if the next mutation followed that suggestion |
| `reasoning_trace` | The LLM's chain-of-thought / scratchpad output |
        """)

    # ── Recommendation ────────────────────────────────────────────────────────
    if recommendation:
        st.info(f"**Recommendation:** {recommendation}")


# ---------------------------------------------------------------------------
# Panel: In-flight Mode
# ---------------------------------------------------------------------------

def _render_inflight(report: dict, df: Optional[pd.DataFrame]) -> None:
    agg = report.get("aggregate_stats", {})
    total_iters = agg.get("total_iterations", len(df) if df is not None else 0)
    best_score = agg.get("best_score", "—")
    experiment_id = report.get("experiment_id", "—")

    st.markdown(f"""
    <div class="status-bar">
      <div class="sb-item"><div class="sb-label">Status</div><div class="sb-val">Post-mortem</div></div>
      <div class="sb-item"><div class="sb-label">Experiment</div><div class="sb-val">{experiment_id}</div></div>
      <div class="sb-item"><div class="sb-label">Iterations</div><div class="sb-val">{total_iters}</div></div>
      <div class="sb-item"><div class="sb-label">Best Score</div><div class="sb-val">{_fmt(best_score, 3)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.info("In-flight mode is for live/active experiments. This report is a completed post-mortem.")

    dimensions = report.get("dimensions", [])
    if dimensions:
        st.markdown('<div class="section-title">Dimension Ratings</div>', unsafe_allow_html=True)
        rows = ""
        for dim in dimensions:
            name = dim.get("name", "Unknown")
            rating = dim.get("rating")
            note = dim.get("summary", "")[:80]
            rows += f"""<tr>
              <td><strong>{name}</strong></td>
              <td>{_stars(rating)}</td>
              <td>{_rating_badge(rating)}</td>
              <td style="font-size:11px">{note}</td>
            </tr>"""
        st.markdown(f"""
        <table class="dim-table">
          <thead><tr><th>Dimension</th><th>Rating</th><th>Status</th><th>Note</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel: Evaluator Metrics Detail
# ---------------------------------------------------------------------------

def _render_sub_metrics(report: dict, df: Optional[pd.DataFrame]) -> None:
    import plotly.graph_objects as go
    from skydiscover.extras.evolve_analyzer.quantitative.sub_metric_analyzer import (
        analyze_sub_metrics,
        _flatten_metrics,
    )

    sub_dim = next(
        (d for d in report.get("dimensions", []) if d.get("name") == "Evaluator Metrics"), None
    )

    sm_analysis = None
    if df is not None and "evaluator_metrics" in df.columns:
        records = df.to_dict(orient="records")
        sm_analysis = analyze_sub_metrics(records)

    if sm_analysis is None or not sm_analysis.metrics:
        msg = sub_dim.get("summary", "No evaluator metric data available.") if sub_dim else "No evaluator metric data available."
        st.info(msg)
        return

    if sub_dim:
        if sub_dim.get("summary"):
            st.markdown(
                f'<div class="insight-box"><strong>Summary:</strong> {sub_dim["summary"]}</div>',
                unsafe_allow_html=True,
            )
        if sub_dim.get("recommendation"):
            st.markdown(
                f'<div class="insight-box">💡 {sub_dim["recommendation"]}</div>',
                unsafe_allow_html=True,
            )

    sorted_stats = sorted(
        sm_analysis.metrics.values(),
        key=lambda s: abs(s.improvement_vs_seed) if s.improvement_vs_seed is not None else 0.0,
        reverse=True,
    )

    st.markdown('<div class="section-title">All Evaluator Metrics</div>', unsafe_allow_html=True)
    rows = ""
    for s in sorted_stats:
        direction_icon = "↓" if s.lower_is_better else "↑"
        is_primary = s.name == sm_analysis.primary_driver
        name_cell = (
            f'<strong style="color:#1a73e8">{s.name} ★</strong>'
            if is_primary
            else s.name
        )
        seed_str = f"{s.seed_value:.4g}" if s.seed_value is not None else "—"
        best_str = f"{s.best:.4g} (iter {s.best_iteration})"
        final_str = f"{s.final:.4g}"
        if s.improvement_vs_seed is not None:
            pct = abs(s.improvement_vs_seed) * 100
            improved = s.improvement_vs_seed > 0
            color = "#137333" if improved else "#d93025"
            sign = "+" if improved else "−"
            imp_str = f'<span style="color:{color};font-weight:600">{sign}{pct:.1f}%</span>'
        else:
            imp_str = "—"
        hl = ' style="background:#e8f0fe"' if is_primary else ""
        rows += f"""<tr{hl}>
          <td>{name_cell}</td>
          <td style="text-align:center;color:#5f6368">{direction_icon}</td>
          <td>{seed_str}</td>
          <td>{best_str}</td>
          <td>{final_str}</td>
          <td>{imp_str}</td>
        </tr>"""
    st.markdown(
        f"""
    <table class="dim-table">
      <thead><tr><th>Metric</th><th>Dir.</th><th>Seed</th><th>Best (iter)</th><th>Final</th><th>vs Seed</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """,
        unsafe_allow_html=True,
    )

    if df is not None and "evaluator_metrics" in df.columns and "iteration" in df.columns:
        metric_series: dict[str, list[tuple[int, float]]] = {}
        for _, row in df.iterrows():
            em = row.get("evaluator_metrics")
            if not isinstance(em, dict):
                continue
            if row.get("child_score") is None:
                continue
            iteration = int(row.get("iteration", 0))
            for name, val in _flatten_metrics(em).items():
                metric_series.setdefault(name, []).append((iteration, val))

        top_metrics = [s.name for s in sorted_stats[:8] if s.name in metric_series]
        if top_metrics:
            st.markdown('<div class="section-title">Metric Trajectories</div>', unsafe_allow_html=True)
            fig = go.Figure()
            palette = ["#1a73e8", "#34a853", "#fbbc04", "#ea4335", "#9c27b0", "#00bcd4", "#ff5722", "#795548"]
            for i, name in enumerate(top_metrics):
                pts = sorted(metric_series[name], key=lambda t: t[0])
                is_primary = name == sm_analysis.primary_driver
                fig.add_trace(go.Scatter(
                    x=[t[0] for t in pts],
                    y=[t[1] for t in pts],
                    mode="lines+markers",
                    name=f"{'★ ' if is_primary else ''}{name}",
                    line=dict(color=palette[i % len(palette)], width=2.5 if is_primary else 1.5,
                              dash="solid" if is_primary else "dot"),
                    marker=dict(size=4),
                ))
            fig.update_layout(
                title="Sub-Metric Trajectories (top 8 by improvement vs seed)",
                xaxis_title="Iteration", yaxis_title="Value",
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
                margin=dict(l=40, r=20, t=40, b=90), height=420,
            )
            fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
            fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Panel: Infrastructure
# ---------------------------------------------------------------------------

def _render_infrastructure(report: dict, df: Optional[pd.DataFrame]) -> None:
    import ast
    import re
    import plotly.graph_objects as go

    infra_dim = next(
        (d for d in report.get("dimensions", []) if d.get("name") == "Infrastructure"),
        None,
    )
    if infra_dim is None:
        st.info("No infrastructure dimension data available.")
        return

    rating = infra_dim.get("rating")
    summary = infra_dim.get("summary", "")
    recommendation = infra_dim.get("recommendation", "")
    evidence = infra_dim.get("evidence", [])

    # --- Status card ---
    badge = _rating_badge(rating)
    st.markdown(
        f'<div class="insight-box">'
        f'<strong>Infrastructure Health {badge}</strong><br>'
        f'{summary}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if recommendation:
        st.markdown(
            f'<div class="pill-row"><div class="pill">💡 {recommendation}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Key metrics pills (non-log evidence lines) ---
    metric_lines = [e for e in evidence if not e.strip().startswith("›")]
    log_lines = [e for e in evidence if e.strip().startswith("›")]

    if metric_lines:
        pills_html = "".join(f'<div class="pill">{e}</div>' for e in metric_lines)
        st.markdown(f'<div class="pill-row">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown("")

    # --- Score progression with sentinel iterations highlighted ---
    if df is not None and "child_score" in df.columns:
        work_df = df.copy()
        if "iteration" not in work_df.columns:
            work_df["iteration"] = range(len(work_df))
        if "best_so_far" not in work_df.columns:
            work_df["best_so_far"] = work_df["child_score"].expanding().max()

        # Parse sentinel iteration numbers from evidence (e.g. "Sentinel iterations: [38, 39, ...]...")
        sentinel_iters: set[int] = set()
        for line in evidence:
            if line.startswith("Sentinel iterations:"):
                m = re.search(r"\[([^\]]+)\]", line)
                if m:
                    try:
                        sentinel_iters = set(ast.literal_eval(f"[{m.group(1)}]"))
                    except (ValueError, SyntaxError):
                        pass
                break

        # Fall back: mark any iteration whose child_score looks like a sentinel value
        if not sentinel_iters:
            sentinel_iters = set(
                work_df.loc[work_df["child_score"] <= -9999, "iteration"].tolist()
            )

        # Parse crash onset from evidence
        crash_onset: Optional[int] = None
        for line in evidence:
            m = re.match(r"Server crash onset at iteration (\d+)", line)
            if m:
                crash_onset = int(m.group(1))
                break

        # Parse degradation window from evidence
        deg_start: Optional[int] = None
        deg_end: Optional[int] = None
        for line in evidence:
            m = re.match(r"Degradation window: iterations (\d+)[–-](\d+)", line)
            if m:
                deg_start, deg_end = int(m.group(1)), int(m.group(2))
                break

        fig = go.Figure()

        # Highlight degradation window
        if deg_start is not None and deg_end is not None:
            fig.add_vrect(
                x0=deg_start, x1=deg_end + 1,
                fillcolor="rgba(249,171,0,0.15)", line_color="rgba(249,171,0,0.5)",
                line_dash="dash",
                annotation_text=f"Degradation {deg_start}–{deg_end}",
                annotation_position="top left",
                annotation_font_size=9,
            )

        # Highlight sentinel region (from crash onset to end)
        if crash_onset is not None:
            fig.add_vrect(
                x0=crash_onset, x1=work_df["iteration"].max(),
                fillcolor="rgba(217,48,37,0.07)", line_color="rgba(217,48,37,0.3)",
                line_dash="dash",
                annotation_text=f"Crash onset iter {crash_onset}",
                annotation_position="top right",
                annotation_font_size=9,
            )

        # Color sentinel points red, healthy points blue
        marker_colors = [
            "#d93025" if it in sentinel_iters else "#8ab4f8"
            for it in work_df["iteration"]
        ]
        marker_sizes = [
            7 if it in sentinel_iters else 4
            for it in work_df["iteration"]
        ]

        fig.add_trace(go.Scatter(
            x=work_df["iteration"], y=work_df["child_score"],
            mode="markers", name="Score (sentinel = red)",
            marker=dict(color=marker_colors, size=marker_sizes, opacity=0.85),
        ))
        fig.add_trace(go.Scatter(
            x=work_df["iteration"], y=work_df["best_so_far"],
            mode="lines", name="Best-so-far",
            line=dict(color="#1a73e8", width=2.5),
        ))

        fig.update_layout(
            title="Score Progression — Sentinel Iterations Highlighted",
            xaxis_title="Iteration", yaxis_title="Score",
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=40, r=20, t=40, b=60), height=370,
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
        fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No iteration DataFrame available for score progression chart.")

    # --- Log evidence ---
    if log_lines:
        st.markdown('<div class="section-title">Log Evidence</div>', unsafe_allow_html=True)
        # Also include the "Log: …" header lines that precede the indented samples
        log_block_lines = [e for e in evidence if e.startswith("Log:") or e.strip().startswith("›")]
        for line in log_block_lines:
            if line.startswith("Log:"):
                st.markdown(f"**{line}**")
            else:
                st.code(line.strip().lstrip("› "), language=None)


# ---------------------------------------------------------------------------
# Sidebar + main entry
# ---------------------------------------------------------------------------

PANEL_NAMES = [
    "Overview", "Score Progression", "Alert Panel", "Failure Budget",
    "Mutation Effectiveness", "Compliance × Score", "Cascade Stages",
    "Evaluator Artifacts", "Efficiency Curve", "Explore vs Exploit",
    "Search Space", "Meta-Analysis", "Evaluator Metrics", "Infrastructure",
]

PANEL_ICONS = ["📊", "📈", "⚠️", "🎯", "🔬", "✅", "📉", "🔍", "💰", "🧭", "🗺️", "🧠", "📐", "🖥️"]

DIMENSION_TO_PANEL: dict[str, str] = {
    "Convergence": "Score Progression",
    "Stagnation": "Alert Panel",
    "Regression": "Alert Panel",
    "Efficiency": "Efficiency Curve",
    "Exploration": "Explore vs Exploit",
    "Search Space": "Search Space",
    "Ceiling": "Score Progression",
    "Meta-Analysis": "Meta-Analysis",
    "Evaluator Metrics": "Evaluator Metrics",
    "Infrastructure": "Infrastructure",
}


def _render_sidebar(report: dict) -> str:
    experiment_id = report.get("experiment_id", "unknown")
    panel_from_url = st.query_params.get("panel", "")
    if panel_from_url in PANEL_NAMES:
        st.session_state["nav_panel"] = panel_from_url
        del st.query_params["panel"]

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:14px 0 12px;border-bottom:1px solid #e0e0e0;margin-bottom:8px">
          <div style="font-size:13px;font-weight:700;color:#1a73e8;letter-spacing:.5px;text-transform:uppercase">EvoExp Analyzer</div>
          <div style="font-size:11px;color:#5f6368;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{experiment_id}</div>
        </div>
        <div style="font-size:10px;font-weight:600;color:#9aa0a6;text-transform:uppercase;letter-spacing:.8px;padding:8px 0 4px">Post-mortem</div>
        """, unsafe_allow_html=True)

        selected = st.radio(
            "Navigation",
            options=PANEL_NAMES,
            label_visibility="collapsed",
            format_func=lambda x: f"{PANEL_ICONS[PANEL_NAMES.index(x)]}  {x}",
            key="nav_panel",
        )

        st.markdown("---")
        st.markdown('<div style="font-size:10px;font-weight:600;color:#9aa0a6;text-transform:uppercase;letter-spacing:.8px;padding:4px 0">Live</div>',
                    unsafe_allow_html=True)

        if st.button("⚡  In-flight Mode", use_container_width=True):
            return "In-flight Mode"

        st.markdown(
            '<div style="font-size:11px;color:#9aa0a6;margin-top:16px">v0.4.1 · Post-mortem</div>',
            unsafe_allow_html=True,
        )
    return selected


def run_dashboard(report_dir: str) -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    report = load_report(report_dir)
    df = load_df(report_dir)

    selected = _render_sidebar(report)

    agg = report.get("aggregate_stats", {})
    total_iters = agg.get("total_iterations", len(df) if df is not None else "?")
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:20px">
      <div>
        <div style="font-size:12px;color:#9aa0a6">EvoExp Analyzer</div>
        <div style="font-size:20px;font-weight:700;color:#202124">{selected}</div>
      </div>
      <div style="flex:1"></div>
      <span style="background:#e6f4ea;color:#137333;font-size:11px;font-weight:600;padding:4px 10px;border-radius:12px;border:1px solid #ceead6">Post-mortem · {total_iters} iters</span>
    </div>
    """, unsafe_allow_html=True)

    dispatch = {
        "Overview": lambda: _render_overview(report, df),
        "Score Progression": lambda: _render_score_progression(report, df),
        "Alert Panel": lambda: _render_alert_panel(report, df),
        "Failure Budget": lambda: _render_failure_budget(report, df),
        "Mutation Effectiveness": lambda: _render_mutation_effectiveness(report, df),
        "Compliance × Score": lambda: _render_compliance_score(report, df),
        "Cascade Stages": lambda: _render_cascade_stages(report, df),
        "Evaluator Artifacts": lambda: _render_artifact_clusters(report),
        "Efficiency Curve": lambda: _render_efficiency_curve(report, df),
        "Explore vs Exploit": lambda: _render_exploration(report, df),
        "Search Space": lambda: _render_search_space(report, df),
        "Meta-Analysis": lambda: _render_meta_analysis(report, df),
        "In-flight Mode": lambda: _render_inflight(report, df),
        "Evaluator Metrics": lambda: _render_sub_metrics(report, df),
        "Infrastructure": lambda: _render_infrastructure(report, df),
    }
    dispatch.get(selected, lambda: st.info("Select a panel from the sidebar."))()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Evolve Loop Analysis Dashboard")
    parser.add_argument("--report-dir", default="./evolve_analysis_output",
                        help="Directory containing report.json and optionally df.parquet / df.csv")
    args, _ = parser.parse_known_args(argv)
    run_dashboard(args.report_dir)
