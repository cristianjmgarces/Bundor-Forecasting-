from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    PLOT_FIGURE_SIZE,
    PLOT_FORECAST_COLOR,
    PLOT_FORECAST_LABEL,
    PLOT_HISTORICAL_COLOR,
    PLOT_HISTORICAL_LABEL,
    PLOT_TITLE_TEMPLATE,
)

logger = logging.getLogger(__name__)


def plot_forecast(
    timeline: pd.DataFrame,
    forecast_df: pd.DataFrame,
    model_name: str,
    selection_info: dict | None = None,
    show_imputed: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PLOT_FIGURE_SIZE)

    n_hist = len(timeline)
    x_hist = np.arange(n_hist)
    y_hist = timeline["suma_de_litros"].to_numpy(dtype=float)

    x_fc = np.arange(n_hist - 1, n_hist + len(forecast_df))
    y_fc_full = np.concatenate([[y_hist[-1]], forecast_df["forecast"].to_numpy()])

    ax.plot(
        x_hist,
        y_hist,
        color=PLOT_HISTORICAL_COLOR,
        linewidth=1.8,
        label=PLOT_HISTORICAL_LABEL,
        zorder=3,
    )

    if show_imputed and "imputed" in timeline.columns:
        imp_mask = timeline["imputed"].to_numpy(dtype=bool)
        if imp_mask.any():
            ax.scatter(
                x_hist[imp_mask],
                y_hist[imp_mask],
                marker="x",
                color="crimson",
                s=40,
                zorder=4,
                label="Semana imputada (= 0)",
            )

    ax.plot(
        x_fc,
        y_fc_full,
        color=PLOT_FORECAST_COLOR,
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=4,
        label=f"{PLOT_FORECAST_LABEL} ({model_name})",
        zorder=3,
    )

    ax.axvspan(
        n_hist - 1,
        n_hist + len(forecast_df) - 1,
        alpha=0.05,
        color=PLOT_FORECAST_COLOR,
        label="_nolegend_",
    )

    ax.axvline(
        x=n_hist - 1,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label="Inicio pronóstico",
    )

    _set_x_ticks(ax, timeline, forecast_df)

    ax.set_ylabel("Suma de litros", fontsize=12)
    ax.set_xlabel("Semana (Año-Semana)", fontsize=12)

    title = PLOT_TITLE_TEMPLATE.format(model_name=model_name)
    if selection_info:
        subtitle_parts = []
        for key in ("variedades", "formatos", "clientes"):
            val = selection_info.get(key)
            if val and val not in ("Todas", "Todos"):
                subtitle_parts.append(f"{key.capitalize()}: {', '.join(map(str, val))}")
        if subtitle_parts:
            title += "\n" + " | ".join(subtitle_parts)

    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    fig.tight_layout()

    logger.info(
        "Gráfico generado: %d obs históricas, %d pasos de pronóstico",
        n_hist,
        len(forecast_df),
    )
    return fig


def _set_x_ticks(
    ax: plt.Axes,
    timeline: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    n_hist = len(timeline)
    n_fc = len(forecast_df)
    total = n_hist + n_fc

    step = max(1, total // 12)
    tick_positions: list[int] = list(range(0, total, step))
    if (total - 1) not in tick_positions:
        tick_positions.append(total - 1)

    labels: list[str] = []
    hist_rows = timeline[["año", "semana"]].values.tolist()
    fc_rows = forecast_df[["año", "semana"]].values.tolist()
    all_rows = hist_rows + fc_rows

    for pos in tick_positions:
        if pos < len(all_rows):
            yr, wk = int(all_rows[pos][0]), int(all_rows[pos][1])
            labels.append(f"{yr}-S{wk:02d}")
        else:
            labels.append("")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)