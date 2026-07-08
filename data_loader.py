from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import pandas as pd

from config import (
    ANIO_MAX,
    ANIO_MIN,
    COLUMN_TYPES,
    REQUIRED_COLUMNS,
    SEMANA_MAX,
    SEMANA_MIN,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    row_count: int = 0
    duplicate_count: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [
            f"Filas leídas : {self.row_count}",
            f"Duplicados   : {self.duplicate_count}",
            f"Estado       : {'OK' if self.is_valid else 'ERRORES CRÍTICOS'}",
        ]
        if self.errors:
            lines.append("Errores críticos:")
            lines.extend(f"  • {e}" for e in self.errors)
        if self.warnings:
            lines.append("Advertencias:")
            lines.extend(f"  • {w}" for w in self.warnings)
        if self.missing_counts:
            lines.append("Valores nulos por columna:")
            for col, cnt in self.missing_counts.items():
                if cnt:
                    lines.append(f"  • {col}: {cnt}")
        return "\n".join(lines)


def load_csv(source: Union[str, Path, IO]) -> tuple[pd.DataFrame, ValidationReport]:
    report = ValidationReport()
    empty = pd.DataFrame()

    import io as _io

    _ENCODINGS = ["utf-8", "latin-1", "cp1252", "utf-8-sig", "iso-8859-1"]
    df = None
    used_encoding = None

    if isinstance(source, _io.StringIO):
        try:
            df = pd.read_csv(source, sep=None, engine="python")            
            used_encoding = "str"
        except Exception as exc:
            report.add_error(f"No se pudo leer el archivo: {exc}")
            return empty, report
    elif hasattr(source, "read"):
        raw_bytes = source.read()
        for enc in _ENCODINGS:
            try:
                df = pd.read_csv(_io.BytesIO(raw_bytes), encoding=enc, sep=None, engine="python")
                used_encoding = enc
                break
            except Exception:
                continue
    else:
        for enc in _ENCODINGS:
            try:
                df = pd.read_csv(source, encoding=enc, sep=None, engine="python")
                used_encoding = enc
                break
            except Exception:
                continue

    if df is None:
        report.add_error(
            "No se pudo leer el archivo (codificación o formato no reconocidos). "
            "Guarda el CSV con codificación UTF-8 desde Excel: "
            "Archivo → Guardar como → CSV UTF-8 (delimitado por comas)."
        )
        return empty, report

    logger.info("CSV leído: %d filas, %d columnas (codificación: %s)", *df.shape, used_encoding)

    report.row_count = len(df)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add_error(f"Columnas faltantes: {missing_cols}")
        return empty, report

    report.missing_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    null_critical = {c: n for c, n in report.missing_counts.items() if n > 0}
    if null_critical:
        report.add_warning(
            f"Valores nulos detectados: "
            + ", ".join(f"{c}={n}" for c, n in null_critical.items())
        )

    def _fix_decimal_comma(value):
        if isinstance(value, str) and "," in value:
            return value.replace(".", "").replace(",", ".")
        return value

    coercion_errors: list[str] = []
    for col, expected_type in COLUMN_TYPES.items():
        try:
            if expected_type == "int":
                df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
            elif expected_type == "float":
                df[col] = df[col].apply(_fix_decimal_comma)
                df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
            elif expected_type == "str":
                df[col] = df[col].astype(str).str.strip()
        except (ValueError, TypeError) as exc:
            coercion_errors.append(f"'{col}': {exc}")

    if coercion_errors:
        report.add_error("Error de tipado en columnas: " + "; ".join(coercion_errors))
        return empty, report

    bad_semana = df[(df["semana"] < SEMANA_MIN) | (df["semana"] > SEMANA_MAX)]
    if not bad_semana.empty:
        report.add_warning(
            f"{len(bad_semana)} fila(s) con semana fuera de [{SEMANA_MIN},{SEMANA_MAX}]"
        )

    bad_anio = df[(df["año"] < ANIO_MIN) | (df["año"] > ANIO_MAX)]
    if not bad_anio.empty:
        report.add_warning(
            f"{len(bad_anio)} fila(s) con año fuera de [{ANIO_MIN},{ANIO_MAX}]"
        )

    dup_keys = ["año", "semana", "variedad", "formato", "cliente"]
    duplicates = df[df.duplicated(subset=dup_keys, keep=False)]
    report.duplicate_count = len(duplicates)
    if report.duplicate_count:
        report.add_warning(
            f"{report.duplicate_count} filas duplicadas (misma clave año+semana+variedad+formato+cliente)"
        )

    logger.info("Validación completada. Válido=%s", report.is_valid)
    return df, report


def get_unique_values(df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "variedad": sorted(df["variedad"].dropna().unique().tolist()),
        "formato": sorted(df["formato"].dropna().unique().tolist()),
        "cliente": sorted(df["cliente"].dropna().unique().tolist()),
    }