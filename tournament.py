from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import MODEL_TIMEOUT_SECONDS
from metrics import compute_all_metrics
from models.arima_model import ARIMAModel
from models.base_model import BaseModel
from models.holt_winters_model import HoltWintersModel
from models.sarima_model import SARIMAModel
from models.spectral_model import SpectralModel

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model: BaseModel | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    error_msg: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error_msg)


@dataclass
class TournamentResult:
    results: list[ModelResult]
    winner: ModelResult
    ranked: list[ModelResult]

    def metrics_table(self) -> list[dict]:
        rows = []
        for r in self.ranked:
            rows.append({
                "Modelo": r.name,
                "Rank": r.rank,
                "SMAPE (%)": round(r.metrics.get("SMAPE", float("nan")), 4),
                "MSE": round(r.metrics.get("MSE", float("nan")), 4),
                "MAPE (%)": round(r.metrics.get("MAPE", float("nan")), 4),
                "Ganador": "✓" if r.name == self.winner.name else "",
            })
        return rows


class ModelTournament:
    def __init__(self, validation_fraction: float = 0.20) -> None:
        if not (0 < validation_fraction < 1):
            raise ValueError("validation_fraction debe estar en (0, 1)")
        self.validation_fraction = validation_fraction

    def run(self, series: np.ndarray) -> TournamentResult:
        series = np.asarray(series, dtype=float)
        n = len(series)

        seasonal_period = 52
        min_train = 2 * seasonal_period

        if n < min_train + 8:
            val_size = max(1, int(n * self.validation_fraction))
        else:
            val_size = min(seasonal_period, n - min_train)

        train = series[: n - val_size]
        val = series[n - val_size :]

        logger.info(
            "Torneo iniciado: %d obs totales, %d entrenamiento, %d validación (%.1f ciclos)",
            n, len(train), len(val), len(val) / seasonal_period,
        )

        candidates: list[BaseModel] = [
            SpectralModel(),
            HoltWintersModel(),
            SARIMAModel(),
            ARIMAModel(),
        ]

        results: list[ModelResult] = []
        for model in candidates:
            result = self._train_and_evaluate(model, train, val)
            results.append(result)
            if result.failed:
                logger.error("Modelo %s falló: %s", model.name, result.error_msg)
            else:
                logger.info(
                    "Modelo %s → SMAPE=%.4f, MSE=%.4f, MAPE=%.4f",
                    model.name,
                    result.metrics["SMAPE"],
                    result.metrics["MSE"],
                    result.metrics["MAPE"],
                )

        successful = [r for r in results if not r.failed]

        if not successful:
            failed_msgs = "; ".join(f"{r.name}: {r.error_msg}" for r in results)
            raise RuntimeError(
                f"Todos los modelos fallaron durante el entrenamiento. Detalles: {failed_msgs}"
            )

        ranked = sorted(
            successful,
            key=lambda r: (
                r.metrics.get("SMAPE", np.inf),
                r.metrics.get("MSE", np.inf),
                r.metrics.get("MAPE", np.inf),
            ),
        )
        for i, r in enumerate(ranked):
            r.rank = i + 1

        winner = ranked[0]
        logger.info("Ganador del torneo: %s (SMAPE=%.4f)", winner.name, winner.metrics["SMAPE"])

        return TournamentResult(results=results, winner=winner, ranked=ranked)

    @staticmethod
    def _train_and_evaluate(
        model: BaseModel,
        train: np.ndarray,
        val: np.ndarray,
    ) -> ModelResult:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

        result = ModelResult(name=model.name, model=model)

        def _fit_and_predict():
            model.fit(train)
            return model.predict(len(val))

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_fit_and_predict)
                y_pred = future.result(timeout=MODEL_TIMEOUT_SECONDS)
            result.metrics = compute_all_metrics(val, y_pred)
            result.params = model.get_params()
        except _Timeout:
            result.error_msg = f"Tiempo límite de {MODEL_TIMEOUT_SECONDS}s superado"
            logger.warning("Modelo %s superó el tiempo límite", model.name)
        except Exception as exc:
            result.error_msg = str(exc)
            logger.exception("Error entrenando %s", model.name)
        return result


def run_tournament(
    series: np.ndarray,
    validation_fraction: float = 0.20,
) -> TournamentResult:
    return ModelTournament(validation_fraction=validation_fraction).run(series)