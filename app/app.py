"""Interface Streamlit para inferência do projeto California Housing."""

from __future__ import annotations

import json
import os
from pathlib import Path

# O backend NumPy permite inferência com modelos Keras sem instalar o pacote
# TensorFlow, que ainda não distribui wheels para todas as versões de Python
# oferecidas pelo Streamlit Community Cloud. Esta variável deve ser definida
# antes de importar o Keras.
os.environ.setdefault("KERAS_BACKEND", "numpy")

import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "melhor_modelo_california_housing.keras"
SCALER_PATH = APP_DIR / "standard_scaler_california_housing.pkl"
PORTABLE_PATH = APP_DIR / "artefatos_portateis.json"
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

FIELDS = {
    "MedInc": ("Renda mediana", "Renda mediana do distrito, em dezenas de milhares de dólares.", 3.5348, 0.0, 20.0),
    "HouseAge": ("Idade das casas", "Idade mediana das residências do distrito, em anos.", 29.0, 0.0, 100.0),
    "AveRooms": ("Média de cômodos", "Número médio de cômodos por domicílio.", 5.2291, 0.0, 150.0),
    "AveBedrms": ("Média de quartos", "Número médio de quartos de dormir por domicílio.", 1.0488, 0.0, 40.0),
    "Population": ("População", "Total de moradores do distrito.", 1166.0, 0.0, 40000.0),
    "AveOccup": ("Ocupação média", "Número médio de moradores por domicílio.", 2.8181, 0.0, 1300.0),
    "Latitude": ("Latitude", "Coordenada geográfica norte-sul do distrito na Califórnia.", 34.26, 32.0, 42.0),
    "Longitude": ("Longitude", "Coordenada geográfica leste-oeste do distrito na Califórnia.", -118.49, -125.0, -113.0),
}


class PortableScaler:
    """Aplica os parâmetros de um StandardScaler exportados no artefato leve."""

    def __init__(self, artifact: dict):
        self.feature_names_in_ = np.asarray(artifact["feature_names_in"])
        self.n_features_in_ = len(self.feature_names_in_)
        self.mean_ = np.asarray(artifact["mean"], dtype=float)
        self.scale_ = np.asarray(artifact["scale"], dtype=float)

    def transform(self, values):
        return (np.asarray(values, dtype=float) - self.mean_) / self.scale_


class PortableLinearModel:
    """Fallback de inferência incluído para o deploy funcionar imediatamente."""

    def __init__(self, artifact: dict):
        self.weights = np.asarray(artifact["weights"], dtype=float)
        self.bias = float(artifact["bias"])

    def predict(self, values, verbose=0):
        del verbose
        return np.asarray(values, dtype=float) @ self.weights[:, None] + self.bias


def load_portable_artifacts(path: Path):
    """Carrega o fallback textual, que pode ser revisado normalmente em PRs."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("format") != "california-housing-portable-v1":
        raise ValueError("Formato dos artefatos portáteis não reconhecido.")
    if artifact.get("feature_names") != FEATURES:
        raise ValueError("As features dos artefatos portáteis não correspondem ao projeto.")
    return PortableLinearModel(artifact["model"]), PortableScaler(artifact["scaler"])


@st.cache_resource(show_spinner="Carregando modelo...")
def load_artifacts():
    """Carrega uma única vez o modelo e o scaler produzidos pelo Notebook 07."""
    model_exists = MODEL_PATH.is_file()
    scaler_exists = SCALER_PATH.is_file()

    if not model_exists and not scaler_exists:
        if not PORTABLE_PATH.is_file():
            raise FileNotFoundError(
                "Nem os artefatos do Notebook 07 nem o fallback textual foram encontrados."
            )
        return load_portable_artifacts(PORTABLE_PATH)

    if model_exists != scaler_exists:
        raise FileNotFoundError(
            "O modelo e o scaler do Notebook 07 devem ser adicionados juntos."
        )

    model = keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)

    scaler_features = list(getattr(scaler, "feature_names_in_", FEATURES))
    if scaler_features != FEATURES:
        raise ValueError(
            "A ordem das features do scaler não corresponde ao Notebook 07: "
            f"{scaler_features}"
        )
    if getattr(scaler, "n_features_in_", len(FEATURES)) != len(FEATURES):
        raise ValueError("O scaler carregado não espera as 8 features do projeto.")
    return model, scaler


def build_input(values: dict[str, float]) -> pd.DataFrame:
    """Cria a entrada na mesma ordem usada pelo notebook durante o treino."""
    return pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)


def main():
    st.set_page_config(
        page_title="Previsão California Housing",
        page_icon="🏡",
        layout="wide",
    )
    st.title("🏡 Previsão de valores — California Housing")
    st.write(
        "Informe as características de um distrito da Califórnia para obter a "
        "estimativa de valor mediano produzida pelo modelo preditivo."
    )

    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as error:
        st.error(f"Não foi possível iniciar a aplicação. {error}")
        st.info(f"Adicione os arquivos do Notebook 07 à pasta `{APP_DIR.name}/` e reinicie.")
        st.stop()
    except Exception as error:
        st.error("Os artefatos existem, mas não puderam ser carregados.")
        st.exception(error)
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Características do distrito")
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(FEATURES):
            label, help_text, default, minimum, maximum = FIELDS[feature]
            with columns[index % 2]:
                values[feature] = st.number_input(
                    f"{label} ({feature})",
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    help=help_text,
                    format="%.4f",
                )
        submitted = st.form_submit_button("Calcular estimativa", type="primary", use_container_width=True)

    if submitted:
        try:
            sample = build_input(values)
            scaled_sample = scaler.transform(sample)
            prediction = float(model.predict(scaled_sample, verbose=0).reshape(-1)[0])
            dollar_value = prediction * 100_000

            st.success("Estimativa calculada com sucesso")
            result, explanation = st.columns([1, 2])
            with result:
                st.metric("Valor mediano estimado", f"US$ {dollar_value:,.2f}")
            with explanation:
                st.write(f"**Saída original do modelo:** `{prediction:.4f}`")
                st.caption(
                    "No California Housing, o alvo MedHouseVal é expresso em "
                    "centenas de milhares de dólares. Por isso, a saída foi "
                    "multiplicada por US$ 100.000."
                )
        except Exception as error:
            st.error("Não foi possível gerar a previsão. Revise os valores informados.")
            st.exception(error)

    st.divider()
    st.warning(
        "Projeto com finalidade exclusivamente educacional. A estimativa representa "
        "distritos do conjunto California Housing e não deve orientar decisões imobiliárias ou financeiras."
    )


if __name__ == "__main__":
    main()
