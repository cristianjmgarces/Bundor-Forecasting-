from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from config import DEFAULT_SEASONAL_PERIOD, MIN_SEASONAL_CYCLES

logger = logging.getLogger(__name__)


class TimeSeriesResult:
    def __init__(
        self,
        timeline: pd.DataFrame,
        selection_info: dict,
    ) -> None:
        self.timeline = timeline.reset_index(drop=True)
        self.series: pd.Series = self.timeline["suma_de_litros"]
        self.n_imputed: int = int(self.timeline["imputed"].sum())
        self.warnings: list[str] = []
        self.selection_info = selection_info

    @property
    def values(self) -> np.ndarray:
        return self.series.to_numpy(dtype=float)

    @property
    def n_obs(self) -> int:
        return len(self.series)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)


def build_time_series(
    df: pd.DataFrame,
    variedades: Sequence[str] | None = None,
    formatos: Sequence[str] | None = None,
    clientes: Sequence[str] | None = None,
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
) -> TimeSeriesResult:
    selection_info: dict = {
        "variedades": list(variedades) if variedades else "Todas",
        "formatos": list(formatos) if formatos else "Todos",
        "clientes": list(clientes) if clientes else "Todos",
    }

    mask = pd.Series([True] * len(df), index=df.index)

    if variedades:
        mask &= df["variedad"].isin(variedades)
    if formatos:
        mask &= df["formato"].isin(formatos)
    if clientes:
        mask &= df["cliente"].isin(clientes)

    filtered = df.loc[mask].copy()

    if filtered.empty:
        raise ValueError(
            "La combinación de filtros seleccionada no produce datos. "
            "Amplíe la selección de variedad, formato o cliente."
        )

    logger.info(
        "Filtrado: %d filas → selección %s", len(filtered), selection_info
    )

    filtered = _redistribute_negatives(filtered)

    aggregated = (
        filtered.groupby(["año", "semana"], as_index=False)["suma_de_litros"]
        .sum()
        .rename(columns={"suma_de_litros": "suma_de_litros"})
    )

    aggregated = _consolidate_week_53(aggregated)

    full_range = _build_full_range(aggregated)

    timeline = full_range.merge(aggregated, on=["año", "semana"], how="left")
    timeline["imputed"] = timeline["suma_de_litros"].isna()
    timeline["suma_de_litros"] = timeline["suma_de_litros"].fillna(0.0)
    timeline = timeline.sort_values(["año", "semana"]).reset_index(drop=True)

    result = TimeSeriesResult(timeline=timeline, selection_info=selection_info)

    n_imputed = result.n_imputed
    n_total = result.n_obs
    if n_imputed > 0:
        pct = 100.0 * n_imputed / n_total
        result.add_warning(
            f"{n_imputed} semana(s) imputada(s) con cero ({pct:.1f}% de la serie)"
        )

    n_cycles = n_total / seasonal_period
    if n_cycles < MIN_SEASONAL_CYCLES:
        result.add_warning(
            f"La serie tiene {n_total} semanas ({n_cycles:.1f} ciclos de {seasonal_period} semanas). "
            f"Se recomienda al menos {MIN_SEASONAL_CYCLES} ciclos completos para un pronóstico fiable."
        )

    logger.info(
        "Serie construida: %d obs, %d imputadas, %.1f ciclos",
        n_total,
        n_imputed,
        n_cycles,
    )
    return result

def _consolidate_week_53(aggregated: pd.DataFrame) -> pd.DataFrame:
    week_53 = aggregated.loc[aggregated["semana"] == 53]
    if week_53.empty:
        return aggregated

    result = aggregated.loc[aggregated["semana"] != 53].copy()
    for _, row in week_53.iterrows():
        anio_siguiente = int(row["año"]) + 1
        volumen = row["suma_de_litros"]
        destino = (result["año"] == anio_siguiente) & (result["semana"] == 1)
        if destino.any():
            result.loc[destino, "suma_de_litros"] += volumen

    return result.sort_values(["año", "semana"]).reset_index(drop=True)

def _build_full_range(aggregated: pd.DataFrame) -> pd.DataFrame:
    anio_min = int(aggregated["año"].min())
    anio_max = int(aggregated["año"].max())

    semana_inicio = int(
        aggregated.loc[aggregated["año"] == anio_min, "semana"].min()
    )

    semana_max_final = int(
        aggregated.loc[aggregated["año"] == anio_max, "semana"].max()
    )

    rows: list[dict] = []
    for anio in range(anio_min, anio_max + 1):
        if anio == anio_min:
            semana_ini = semana_inicio
        else:
            semana_ini = 1

        if anio == anio_max:
            semana_fin = semana_max_final
        else:
            semana_fin = _max_week_in_year(anio)

        for semana in range(semana_ini, semana_fin + 1):
            rows.append({"año": anio, "semana": semana})

    return pd.DataFrame(rows)


def _redistribute_negatives(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(
        ["variedad", "formato", "cliente", "año", "semana"]
    ).reset_index(drop=True)

    for _, group in df.groupby(["variedad", "formato", "cliente"], sort=False):
        indices = group.index.tolist()
        for i, idx in enumerate(indices):
            if df.loc[idx, "suma_de_litros"] < 0:
                neg_val = df.loc[idx, "suma_de_litros"]
                if i > 0:
                    prev_idx = indices[i - 1]
                    df.loc[prev_idx, "suma_de_litros"] = max(
                        0.0, df.loc[prev_idx, "suma_de_litros"] + neg_val
                    )
                df.loc[idx, "suma_de_litros"] = 0.0

    return df


def _max_week_in_year(year: int) -> int:
    import datetime
    dec_28 = datetime.date(year, 12, 28)
    return dec_28.isocalendar()[1]