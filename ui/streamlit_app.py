from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL
from data_loader import ValidationReport, get_unique_values, load_csv
from data_preprocessor import TimeSeriesResult, build_time_series
from forecaster import ForecastResult, generate_forecast
from tournament import TournamentResult, run_tournament
from visualization import plot_forecast

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Bundor Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<style>[data-testid='stStatusWidget']{display:none;}</style>", unsafe_allow_html=True)

_DEFAULTS: dict = {
    "df": None,
    "report": None,
    "ts_result": None,
    "tournament_result": None,
    "forecast_result": None,
}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.title("📂 Cargar datos")
    uploaded_file = st.file_uploader(
        label="Seleccionar archivo CSV",
        type=["csv"],
        help="El CSV debe contener: año, semana, variedad, formato, cliente, suma_de_litros",
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("_file_id") != file_id:
            with st.spinner("Validando archivo…"):
                df, report = load_csv(uploaded_file)
            st.session_state["df"] = df if report.is_valid else None
            st.session_state["report"] = report
            st.session_state["ts_result"] = None
            st.session_state["tournament_result"] = None
            st.session_state["forecast_result"] = None
            st.session_state["_file_id"] = file_id


st.title("Bundor Forecasting")
st.markdown(
    "Herramienta de pronóstico de series temporales basada en un "
    "**sistema de torneo de modelos** (Análisis Espectral · Holt-Winters · SARIMA · ARIMA)."
)

st.header("1. Validación de Datos")

report: ValidationReport | None = st.session_state["report"]
df: pd.DataFrame | None = st.session_state["df"]

if report is None:
    st.info("Carga un archivo CSV desde la barra lateral para comenzar.")
    st.stop()

if df is not None and not df.empty:
    st.subheader("Vista previa del archivo")
    st.dataframe(df.head(5), use_container_width=True)
else:
    st.error("El archivo no pudo cargarse correctamente.")

with st.expander("📋 Reporte de validación completo", expanded=not report.is_valid):
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Filas leídas", report.row_count)
    col_b.metric("Duplicados", report.duplicate_count)
    col_c.metric("Estado", "✅ Válido" if report.is_valid else "❌ Errores")

    if report.errors:
        for err in report.errors:
            st.error(f"**Error:** {err}")

    if report.warnings:
        for warn in report.warnings:
            st.warning(f"**Advertencia:** {warn}")

    if report.missing_counts:
        missing_df = pd.DataFrame(
            [{"Columna": k, "Nulos": v} for k, v in report.missing_counts.items() if v > 0]
        )
        if not missing_df.empty:
            st.subheader("Valores nulos por columna")
            st.dataframe(missing_df, use_container_width=True)

if not report.is_valid or df is None or df.empty:
    st.error("Corrija los errores en el archivo CSV para continuar.")
    st.stop()

st.header("2. Selección del Objetivo de Pronóstico")

unique_vals = get_unique_values(df)

col1, col2, col3 = st.columns(3)
with col1:
    sel_variedades = st.multiselect(
        "Variedad(es)",
        options=unique_vals["variedad"],
        default=[],
        help="Dejar vacío = todas las variedades",
    )
with col2:
    sel_formatos = st.multiselect(
        "Formato(s)",
        options=unique_vals["formato"],
        default=[],
        help="Dejar vacío = todos los formatos",
    )
with col3:
    sel_clientes = st.multiselect(
        "Cliente(s)",
        options=unique_vals["cliente"],
        default=[],
        help="Dejar vacío = todos los clientes",
    )

if st.button("Cargar serie temporal", type="secondary"):
    with st.spinner("Construyendo serie temporal…"):
        try:
            ts_result = build_time_series(
                df,
                variedades=sel_variedades or None,
                formatos=sel_formatos or None,
                clientes=sel_clientes or None,
            )
            st.session_state["ts_result"] = ts_result
            st.session_state["tournament_result"] = None
            st.session_state["forecast_result"] = None
        except ValueError as exc:
            st.error(str(exc))

ts_result: TimeSeriesResult | None = st.session_state["ts_result"]

if ts_result is not None:
    for warn in ts_result.warnings:
        st.warning(warn)

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Observaciones totales", ts_result.n_obs)
    col_info2.metric("Semanas imputadas (= 0)", ts_result.n_imputed)
    col_info3.metric(
        "Pct. imputado",
        f"{100 * ts_result.n_imputed / ts_result.n_obs:.1f} %",
    )

    with st.expander("📊 Serie histórica completa (con imputaciones)", expanded=True):
        display_df = ts_result.timeline[["año", "semana", "suma_de_litros", "imputed"]].copy()
        display_df.columns = ["Año", "Semana", "Litros", "Imputado"]
        st.dataframe(display_df, use_container_width=True, height=250)

st.header("3. Configuración del Pronóstico")

horizon = st.number_input(
    "Horizonte de pronóstico (semanas)",
    min_value=1,
    max_value=104,
    value=12,
    step=1,
    help="Número de semanas futuras a pronosticar (1–104)",
)

run_disabled = ts_result is None
run_clicked = st.button(
    "Ejecutar Pronóstico",
    type="primary",
    disabled=run_disabled,
)

if run_disabled:
    st.caption("Primero carga y confirma la serie temporal (Sección 2).")

if run_clicked and ts_result is not None:
    progress = st.progress(0, text="Iniciando torneo de modelos…")

    
    try:
        progress.progress(10, text="Entrenando Análisis Espectral…")
        tournament_result = run_tournament(ts_result.values, validation_fraction=0.20)
        progress.progress(80, text="Generando pronóstico final…")
        forecast_result = generate_forecast(
            tournament_result=tournament_result,
            series=ts_result.values,
            timeline=ts_result.timeline,
            horizon=int(horizon),
        )
        st.session_state["tournament_result"] = tournament_result
        st.session_state["forecast_result"] = forecast_result
        progress.progress(100, text="Completado ✓")
    except RuntimeError as exc:
        st.error(f"Error durante el pronóstico: {exc}")
    except Exception as exc:
        st.error(f"Error inesperado: {exc}")
        logger.exception("Error inesperado en la ejecución del pronóstico")

tournament_result: TournamentResult | None = st.session_state["tournament_result"]
forecast_result: ForecastResult | None = st.session_state["forecast_result"]

if tournament_result is None or forecast_result is None:
    st.stop()

st.header("4. Resultados del Torneo y Pronóstico")

st.subheader("A. Tabla Comparativa de Métricas")

metrics_rows = tournament_result.metrics_table()
metrics_df = pd.DataFrame(metrics_rows)

def _highlight_winner(row: pd.Series) -> list[str]:
    return [
        "background-color: #d4edda; font-weight: bold" if row["Ganador"] == "✓" else ""
    ] * len(row)

st.dataframe(
    metrics_df.style.apply(_highlight_winner, axis=1),
    use_container_width=True,
)

failed = [r for r in tournament_result.results if r.failed]
if failed:
    with st.expander("⚠️ Modelos que fallaron durante el entrenamiento"):
        for r in failed:
            st.error(f"**{r.name}**: {r.error_msg}")

st.subheader("B. Modelo Ganador")

winner_result = tournament_result.winner

col_w1, col_w2 = st.columns([1, 2])
with col_w1:
    st.success(f"**{winner_result.name}**")
    st.metric("SMAPE", f"{winner_result.metrics['SMAPE']:.4f} %")
    st.metric("MSE", f"{winner_result.metrics['MSE']:.4f}")
    mape_val = winner_result.metrics.get("MAPE", float("nan"))
    st.metric("MAPE", f"{mape_val:.4f} %" if not pd.isna(mape_val) else "N/A")

with col_w2:
    with st.expander("Parámetros del modelo ganador", expanded=False):
        params_df = pd.DataFrame(
            [{"Parámetro": k, "Valor": str(v)} for k, v in forecast_result.model_params.items()
             if v is not None]
        )
        if not params_df.empty:
            st.dataframe(params_df, use_container_width=True)

    st.markdown("**Ranking completo**")
    ranking_rows = [
        {
            "Pos.": r.rank,
            "Modelo": r.name,
            "SMAPE (%)": f"{r.metrics.get('SMAPE', float('nan')):.4f}",
            "Ganador": "✓" if r.name == winner_result.name else "",
        }
        for r in tournament_result.ranked
    ]
    st.dataframe(pd.DataFrame(ranking_rows), use_container_width=True)

st.subheader("C. Gráfico del Pronóstico")

fig = plot_forecast(
    timeline=ts_result.timeline,
    forecast_df=forecast_result.forecast_df,
    model_name=forecast_result.model_name,
    selection_info=ts_result.selection_info,
    show_imputed=True,
)
st.pyplot(fig, use_container_width=True)

with st.expander("📋 Valores del pronóstico (tabla)"):
    fc_display = forecast_result.forecast_df.copy()
    fc_display.columns = ["Año", "Semana", "Pronóstico (litros)", "Paso"]
    fc_display["Pronóstico (litros)"] = fc_display["Pronóstico (litros)"].round(2)
    st.dataframe(fc_display, use_container_width=True)

    csv_bytes = fc_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar pronóstico CSV",
        data=csv_bytes,
        file_name=f"pronostico_{forecast_result.model_name}_{int(horizon)}semanas.csv",
        mime="text/csv",
    )