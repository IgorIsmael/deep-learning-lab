"""Interface Streamlit para inferência do projeto California Housing."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "melhor_modelo_california_housing.keras"
SCALER_PATH = APP_DIR / "standard_scaler_california_housing.pkl"
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


@st.cache_resource(show_spinner="Carregando modelo...")
def load_artifacts():
    """Carrega uma única vez o modelo e o scaler produzidos pelo Notebook 07."""
    missing = [path.name for path in (MODEL_PATH, SCALER_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Artefato(s) não encontrado(s): " + ", ".join(missing)
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
        "estimativa de valor mediano produzida pela rede neural."
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
