from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, series: np.ndarray) -> "BaseModel":
        """Entrenar el modelo con la serie histórica y devolver self."""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Generar un array 1D de longitud horizon con el pronóstico."""

    @abstractmethod
    def fitted_values(self) -> np.ndarray:
        """Valores ajustados sobre el conjunto de entrenamiento."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Parámetros del modelo entrenado."""

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"El modelo '{self.name}' debe entrenarse antes de llamar a este método."
            )

    def __repr__(self) -> str:
        status = "entrenado" if self._is_fitted else "no entrenado"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"