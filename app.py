import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap

# 📁 Caminhos seguros para os arquivos
dir_path = os.path.dirname(os.path.abspath(__file__))

# Carregar modelos e scaler
model_mrr = joblib.load(os.path.join(dir_path, "modelo_mrr.pkl"))
model_churn = joblib.load(os.path.join(dir_path, "modelo_churn.pkl"))
scaler = joblib.load(os.path.join(dir_path, "scaler.pkl"))

# Título do app
st.set_page_config(page_title="Oráculo Preditivo", layout="wide")
st.title("Oráculo, a previsão de MRR e Churn")
st.markdown("Com base nos leading indicators do negócio, rodamos previsão para o churn e MRR nos próximos 14 dias.")

# Carregar base e últimos dados
df = pd.read_csv(os.path.join(dir_path, "base_modelo_diario.csv"), parse_dates=['Data'])
ultimos_dados = df.sort_values(by='Data').iloc[-1:]
features = ultimos_dados.drop(columns=["Data", "MRR_Total", "Churn_Total"])
X_scaled = scaler.transform(features)

# Previsões
pred_mrr = model_mrr.predict(X_scaled)[0]
pred_churn = model_churn.predict(X_scaled)[0]

# Delta vs média dos últimos 14 dias
media_mrr_14d = df.tail(14)["MRR_Total"].mean()
media_churn_14d = df.tail(14)["Churn_Total"].mean()
delta_mrr = (pred_mrr - media_mrr_14d) / media_mrr_14d * 100
delta_churn = (pred_churn - media_churn_14d) / media_churn_14d * 100

# Métricas principais
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="💰 MRR previsto (14 dias)",
        value=f"R$ {pred_mrr:,.0f}",
        delta=f"{delta_mrr:.2f}%",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="📉 Churn previsto (14 dias)",
        value=f"{pred_churn:.2f}%",
        delta=f"{delta_churn:.2f}%",
        delta_color="inverse"
    )

st.divider()

# Explicabilidade: SHAP - MRR
st.markdown("### As principais influências na previsão de MRR")
explainer_mrr = shap.Explainer(model_mrr)
shap_values_mrr = explainer_mrr(X_scaled)
top_idx_mrr = np.argsort(-np.abs(shap_values_mrr.values[0]))[:3]
for feat, val in zip(features.columns[top_idx_mrr], shap_values_mrr.values[0][top_idx_mrr]):
    sinal = "↑" if val > 0 else "↓"
    st.markdown(f"- **{feat}**: {sinal} impacto de {abs(val):.2f}")

# Explicabilidade: SHAP - Churn
st.markdown("### As principais influências na previsão de Churn")
explainer_churn = shap.Explainer(model_churn)
shap_values_churn = explainer_churn(X_scaled)
top_idx_churn = np.argsort(-np.abs(shap_values_churn.values[0]))[:3]
for feat, val in zip(features.columns[top_idx_churn], shap_values_churn.values[0][top_idx_churn]):
    sinal = "↑" if val > 0 else "↓"
    st.markdown(f"- **{feat}**: {sinal} impacto de {abs(val):.2f}")

st.divider()

# Qualidade dos modelos
st.markdown("### Qualidade dos modelos")
st.markdown("- **MRR** → MAE: R$29.122 | R²: 0.988")
st.markdown("- **Churn** → MAE: 0.049 p.p. | R²: 0.995")

st.divider()

# Mostrar features reais de entrada
st.markdown("### Contexto dos indicadores de entrada")
st.dataframe(features.T.rename(columns={features.index[0]: "Valor"}))