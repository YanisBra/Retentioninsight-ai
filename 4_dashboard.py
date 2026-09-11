"""
4_dashboard.py — Dashboard Interactif de Prédiction du Churn

Projet  : RetentionInsight AI
Auteur  : Équipe Data Science

Description :
Ce dashboard répond à l'exigence du brief projet.md : 
"Création de dashboards: utilisez Plotly ou Bokeh pour créer des visualisations interactives 
permettant d'explorer les prédictions de churn."

Technologies : Streamlit (Interface) + Plotly (Visualisations interactives)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ─────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RetentionInsight AI — Dashboard",
    page_icon="🔮",
    layout="wide",
)

# ─────────────────────────────────────────────
# Chargement des données et du modèle
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # Traitement rapide pour l'affichage brut et le scaler
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    return df

@st.cache_resource
def load_model_and_metadata():
    model_path = "model/churn_model.pkl"
    features_path = "model/feature_names.json"
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        return None, None
        
    pipeline = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)
        
    return pipeline, feature_names

df = load_data()
model, feature_names = load_model_and_metadata()

# ─────────────────────────────────────────────
# En-tête
# ─────────────────────────────────────────────
st.title("🔮 RetentionInsight AI — Dashboard Interactif")
st.markdown("Ce dashboard interactif permet d'explorer les données de résiliation (Churn) et de simuler des prédictions grâce à notre modèle Random Forest finalisé.")

# ─────────────────────────────────────────────
# Section 1 : Visualisations avec Plotly
# ─────────────────────────────────────────────
st.header("📊 Explorateur de Données (Plotly)")

col1, col2, col3 = st.columns(3)

with col1:
    fig_churn = px.pie(
        df, names='Churn', 
        title="Répartition Globale du Churn",
        color='Churn', color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
        hole=0.4
    )
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    fig_contract = px.histogram(
        df, x='Contract', color='Churn', 
        title="Churn par Type de Contrat",
        barmode='group',
        color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"}
    )
    st.plotly_chart(fig_contract, use_container_width=True)

with col3:
    fig_charges = px.histogram(
        df, x='MonthlyCharges', color='Churn', 
        title="Distribution des Charges Mensuelles",
        nbins=20, opacity=0.7,
        color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"}
    )
    st.plotly_chart(fig_charges, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# Section 2 : Simulateur de Prédiction
# ─────────────────────────────────────────────
st.header("⚙️ Simulateur de Prédiction de Churn")
st.markdown("Ajustez les caractéristiques du client ci-dessous pour explorer comment le modèle de Machine Learning évalue son risque de résiliation.")

if model is None:
    st.error("⚠️ Modèle introuvable. Veuillez exécuter `save_model.py` d'abord.")
else:
    # Création du formulaire de simulation
    st.subheader("Définition du Profil Client")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)
        monthly_charges = st.number_input("Charges mensuelles ($)", 15.0, 120.0, 70.0)
        contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
        senior = st.selectbox("Est un senior (âgé)", ["Non", "Oui"])
        
    with c2:
        internet = st.selectbox("Service Internet", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Méthode de paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        partner = st.selectbox("A un partenaire", ["Non", "Oui"])
        dependents = st.selectbox("Personnes à charge", ["Non", "Oui"])
        
    with c3:
        paperless = st.selectbox("Facturation sans papier", ["Oui", "Non"])
        tech_support = st.selectbox("Support technique", ["Non", "Oui", "Pas de service internet"])
        online_security = st.selectbox("Sécurité en ligne", ["Non", "Oui", "Pas de service internet"])
        
    # Bouton de prédiction
    if st.button("🔮 Calculer le risque de résiliation", type="primary"):
        # Calcul simpliste des charges totales pour la démo
        total_charges = tenure * monthly_charges
        
        # Mappage des entrées de l'interface vers le format du modèle (30 colonnes exactes)
        input_dict = {
            "SeniorCitizen": 1 if senior == "Oui" else 0,
            "tenure": float(tenure),
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "gender_Male": 0, # Valeur fixe pour la démo (femme par defaut)
            "Partner_Yes": 1 if partner == "Oui" else 0,
            "Dependents_Yes": 1 if dependents == "Oui" else 0,
            "PhoneService_Yes": 1, # Service téléphonique actif par défaut
            "MultipleLines_No phone service": 0,
            "MultipleLines_Yes": 1,
            "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
            "InternetService_No": 1 if internet == "No" else 0,
            "OnlineSecurity_No internet service": 1 if online_security == "Pas de service internet" or internet == "No" else 0,
            "OnlineSecurity_Yes": 1 if online_security == "Oui" else 0,
            "OnlineBackup_No internet service": 1 if internet == "No" else 0,
            "OnlineBackup_Yes": 0,
            "DeviceProtection_No internet service": 1 if internet == "No" else 0,
            "DeviceProtection_Yes": 0,
            "TechSupport_No internet service": 1 if tech_support == "Pas de service internet" or internet == "No" else 0,
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
        
        try:
            # Construction du vecteur final dans le bon ordre sous forme de DataFrame (attendu par le preprocessor)
            df_pred = pd.DataFrame([input_dict])
            df_pred = df_pred[feature_names]
            
            # Prédiction du modèle
            proba = model.predict_proba(df_pred)[0][1]
            
            st.markdown("---")
            st.subheader("Résultat du Modèle d'Intelligence Artificielle")
            
            if proba > 0.65:
                st.error(f"⚠️ **Risque ÉLEVÉ** de résiliation détecté ! Probabilité : **{proba*100:.1f}%**")
                st.info("💡 **Recommandation :** Contactez ce client immédiatement pour lui proposer une offre de fidélisation.")
            elif proba > 0.35:
                st.warning(f"⚡ **Risque MODÉRÉ** de résiliation. Probabilité : **{proba*100:.1f}%**")
                st.info("💡 **Recommandation :** Surveillez ce client et incluez-le dans la prochaine campagne de marketing préventif.")
            else:
                st.success(f"✅ **Risque FAIBLE** de résiliation. Probabilité : **{proba*100:.1f}%**")
                st.info("💡 **Recommandation :** Ce client est fidèle. Aucune action de rétention n'est requise dans l'immédiat.")
                
        except Exception as e:
            st.error(f"Erreur de traitement lors de la prédiction : {e}")
