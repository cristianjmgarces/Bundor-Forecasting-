from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.arima_model import ARIMAModel
from models.base_model import BaseModel
from models.holt_winters_model import HoltWintersModel
from models.sarima_model import SARIMAModel
from models.spectral_model import SpectralModel
from tournament import TournamentResult

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "SpectralAnalysis": SpectralModel,
    "HoltWinters": HoltWintersModel,
    "SARIMA": SARIMAModel,
    "ARIMA": ARIMAModel,
}


@dataclass
class ForecastResult:
    forecast_values: np.ndarray
    forecast_df: pd.DataFrame
    model_name: str
    model_params: dict
    horizon: int


def generate_forecast(
    tournament_result: TournamentResult,
    series: np.ndarray,
    timeline: pd.DataFrame,
    horizon: int,
) -> ForecastResult:
    if horizon < 1:
        raise ValueError(f"El horizonte debe ser ≥ 1, recibido: {horizon}")

    winner_name = tournament_result.winner.name

    if winner_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Modelo '{winner_name}' no encontrado en el registro. "
            f"Disponibles: {list(_MODEL_REGISTRY)}"
        )

    model_cls = _MODEL_REGISTRY[winner_name]
    model = model_cls()

    logger.info(
        "Re-entrenando modelo ganador '%s' sobre %d observaciones completas",
        winner_name,
        len(series),
    )

    try:
        model.fit(np.asarray(series, dtype=float))
        forecast_values = model.predict(horizon)
        params = model.get_params()
    except Exception as exc:
        raise RuntimeError(
            f"El modelo ganador '{winner_name}' falló al re-entrenarse: {exc}"
        ) from exc

    forecast_df = _build_forecast_timeline(timeline, horizon, forecast_values)

    logger.info(
        "Pronóstico generado: %d semanas, modelo='%s'", horizon, winner_name
    )

    return ForecastResult(
        forecast_values=forecast_values,
        forecast_df=forecast_df,
        model_name=winner_name,
        model_params=params,
        horizon=horizon,
    )


def _build_forecast_timeline(
    timeline: pd.DataFrame,
    horizon: int,
    forecast_values: np.ndarray,
) -> pd.DataFrame:
    last_row = timeline.iloc[-1]
    current_year = int(last_row["año"])
    current_week = int(last_row["semana"])

    rows = []
    for step in range(1, horizon + 1):
        current_week += 1
        max_week = _iso_max_week(current_year)
        if current_week > max_week:
            current_year += 1
            current_week = 1
        rows.append({
            "año": current_year,
            "semana": current_week,
            "forecast": float(forecast_values[step - 1]),
            "step": step,
        })

    return pd.DataFrame(rows)


def generate_all_forecasts(
    tournament_result: TournamentResult,
    series: np.ndarray,
    timeline: pd.DataFrame,
    horizon: int,
) -> dict[str, ForecastResult]:
    results: dict[str, ForecastResult] = {}
    for model_result in tournament_result.ranked:
        try:
            model_cls = _MODEL_REGISTRY[model_result.name]
            model = model_cls()
            model.fit(np.asarray(series, dtype=float))
            forecast_values = model.predict(horizon)
            forecast_df = _build_forecast_timeline(timeline, horizon, forecast_values)
            results[model_result.name] = ForecastResult(
                forecast_values=forecast_values,
                forecast_df=forecast_df,
                model_name=model_result.name,
                model_params=model.get_params(),
                horizon=horizon,
            )
            logger.info("Pronóstico generado para modelo '%s'", model_result.name)
        except Exception as exc:
            logger.warning("No se pudo generar pronóstico para '%s': %s", model_result.name, exc)
    return results


def _iso_max_week(year: int) -> int:
    import datetime
    return datetime.date(year, 12, 28).isocalendar()[1]