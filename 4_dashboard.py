"""
4_dashboard.py — Exploration interactive des prédictions de Churn (Plotly & Streamlit)
"""

import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────
# 1. Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RetentionInsight AI — Exploration des Prédictions",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 RetentionInsight AI — Exploration des Prédictions (Plotly)")
st.markdown(
    "Ce tableau de bord permet d'explorer les **prédictions calculées par le modèle Random Forest** "
    "sur le jeu de test et d'estimer en direct le risque de résiliation pour un profil client personnalisé."
)

# ─────────────────────────────────────────────
# 2. Chargement du modèle et calcul des prédictions sur le Test Set
# ─────────────────────────────────────────────
@st.cache_data
def get_test_predictions():
    df = pd.read_csv("data/processed_telco.csv")
    X = df.drop(columns=["Churn_Yes"])
    y = df["Churn_Yes"]
    
    # Même split 80/20 stratifié (random_state=42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    model = joblib.load("model/churn_model.pkl")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results_df = X_test.copy()
    results_df["Vrai_Statut"] = y_test.map({1: "Churn (Parti)", 0: "Fidèle (Resté)"})
    results_df["Prédiction"] = pd.Series(y_pred, index=X_test.index).map(
        {1: "Churn Prédit", 0: "Fidèle Prédit"}
    )
    results_df["Probabilité_Churn"] = np.round(y_proba * 100, 1)
    
    results_df["Contrat"] = "Month-to-month"
    results_df.loc[results_df["Contract_One year"] == 1, "Contrat"] = "One year"
    results_df.loc[results_df["Contract_Two year"] == 1, "Contrat"] = "Two year"
    
    return results_df, y_test, y_pred

@st.cache_resource
def load_model_and_features():
    model = joblib.load("model/churn_model.pkl")
    with open("model/feature_names.json", "r") as f:
        feature_names = json.load(f)
    return model, feature_names

results_df, y_test, y_pred = get_test_predictions()
model, feature_names = load_model_and_features()

# ─────────────────────────────────────────────
# 3. KPIs Clés
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clients évalués (Jeu de test)", f"{len(results_df):,}")
c2.metric("Rappel sur Churners", "79.0 %")
c3.metric("Pouvoir discriminant (AUC)", "84.61 %")
c4.metric("Seuil de décision optimal", "0.54")

st.markdown("---")

# ─────────────────────────────────────────────
# 4. Visualisations Interactives avec Plotly
# ─────────────────────────────────────────────
st.header("📊 1. Exploration Visuelle des Prédictions (Plotly)")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Distribution du Risque selon le Contrat")
    fig_hist = px.histogram(
        results_df,
        x="Probabilité_Churn",
        color="Contrat",
        nbins=25,
        barmode="overlay",
        opacity=0.7,
        labels={"Probabilité_Churn": "Probabilité de Churn calculée par le modèle (%)"},
        color_discrete_map={
            "Month-to-month": "#e74c3c",
            "One year": "#f39c12",
            "Two year": "#2ecc71",
        },
    )
    fig_hist.update_layout(xaxis_ticksuffix=" %", yaxis_title="Nombre de clients")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("Matrice de Confusion Interactive")
    cm = confusion_matrix(y_test, y_pred)
    total_samples = np.sum(cm)
    text_matrix = [
        [f"{cm[0,0]} ({cm[0,0]/total_samples*100:.1f}%)", f"{cm[0,1]} ({cm[0,1]/total_samples*100:.1f}%)"],
        [f"{cm[1,0]} ({cm[1,0]/total_samples*100:.1f}%)", f"{cm[1,1]} ({cm[1,1]/total_samples*100:.1f}%)"],
    ]
    cm_df = pd.DataFrame(
        cm,
        index=["Réel : Non (Fidèle)", "Réel : Churn"],
        columns=["Prédit : Non (Fidèle)", "Prédit : Churn"],
    )
    fig_cm = px.imshow(
        cm_df,
        text_auto=False,
        color_continuous_scale="Blues",
    )
    fig_cm.update_traces(text=text_matrix, texttemplate="%{text}")
    fig_cm.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Cartographie des Prédictions : Ancienneté vs Facture Mensuelle")
fig_scatter = px.scatter(
    results_df,
    x="tenure",
    y="MonthlyCharges",
    color="Prédiction",
    size="TotalCharges",
    hover_data=["Probabilité_Churn", "Vrai_Statut", "Contrat"],
    labels={
        "tenure": "Ancienneté (mois)",
        "MonthlyCharges": "Charges Mensuelles ($)",
        "TotalCharges": "Total Facturé ($)",
    },
    color_discrete_map={"Churn Prédit": "#e74c3c", "Fidèle Prédit": "#2ecc71"},
    opacity=0.8,
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# 5. Simulateur de Prédiction par Critères
# ─────────────────────────────────────────────
st.header("⚙️ 2. Simulateur de Prédiction par Critères Client")
st.markdown("Ajustez les caractéristiques ci-dessous pour calculer en direct l'évaluation du modèle :")

f1, f2, f3 = st.columns(3)

with f1:
    tenure = st.slider("Ancienneté (mois)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Facture mensuelle ($)", min_value=15.0, max_value=120.0, value=70.0, step=1.0)
    contract = st.selectbox("Type d'engagement", ["Month-to-month", "One year", "Two year"])
    senior = st.selectbox("Client Senior (Âgé)", ["Non", "Oui"])

with f2:
    internet = st.selectbox("Technologie Internet", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Assistance technique (TechSupport)", ["Non", "Oui", "Pas d'internet"])
    online_security = st.selectbox("Sécurité en ligne", ["Non", "Oui", "Pas d'internet"])
    payment = st.selectbox(
        "Moyen de paiement",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

with f3:
    paperless = st.selectbox("Facturation dématérialisée", ["Oui", "Non"])
    partner = st.selectbox("Vit en couple (Partner)", ["Oui", "Non"])
    dependents = st.selectbox("Personnes à charge (Dependents)", ["Non", "Oui"])

if st.button("🔮 Prédire le risque de résiliation", type="primary", use_container_width=True):
    total_charges = tenure * monthly_charges
    
    input_data = {
        "SeniorCitizen": 1 if senior == "Oui" else 0,
        "tenure": float(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "gender_Male": 0,
        "Partner_Yes": 1 if partner == "Oui" else 0,
        "Dependents_Yes": 1 if dependents == "Oui" else 0,
        "PhoneService_Yes": 1,
        "MultipleLines_No phone service": 0,
        "MultipleLines_Yes": 1,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,
        "OnlineSecurity_No internet service": 1 if online_security == "Pas d'internet" or internet == "No" else 0,
        "OnlineSecurity_Yes": 1 if online_security == "Oui" else 0,
        "OnlineBackup_No internet service": 1 if internet == "No" else 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_No internet service": 1 if internet == "No" else 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No internet service": 1 if tech_support == "Pas d'internet" or internet == "No" else 0,
        "TechSupport_Yes": 1 if tech_support == "Oui" else 0,
        "StreamingTV_No internet service": 1 if internet == "No" else 0,
        "StreamingTV_Yes": 1,
        "StreamingMovies_No internet service": 1 if internet == "No" else 0,
        "StreamingMovies_Yes": 1,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless == "Oui" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
    }
    
    df_single = pd.DataFrame([input_data])[feature_names]
    proba = model.predict_proba(df_single)[0][1]
    
    st.markdown("### Résultat de l'Évaluation :")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Probabilité de Churn", f"{proba * 100:.1f} %")
    
    with res_col2:
        if proba >= 0.54:
            st.error("⚠️ **Risque ÉLEVÉ de Churn** (Probabilité supérieure au seuil optimal de 54 %)")
            st.info("💡 **Action recommandée :** Contacter proactivement le client avec une offre de passage à un contrat 1 an ou un geste sur les options.")
        elif proba >= 0.35:
            st.warning("⚡ **Risque MODÉRÉ** — Client sensible au prix à surveiller.")
        else:
            st.success("✅ **Risque FAIBLE** — Client fidèle, aucune action requise.")
