# RetentionInsight AI 🔮

> Prédiction du churn client par Machine Learning — Projet #3  
> Directeur de projet en IA · Année 1 · L'École Multimédia

---

## Présentation

**RetentionInsight AI** est un pipeline complet de Machine Learning pour prédire la résiliation d'abonnement (*churn*) de clients d'une entreprise de télécommunications, basé sur le dataset public **Telco Customer Churn** (Kaggle).

Le projet couvre l'intégralité du cycle de vie ML :
- Exploration et audit de qualité des données (EDA)
- Nettoyage, encodage et optimisation de la mémoire RAM
- Modélisation, optimisation d'hyperparamètres et validation croisée sans fuite
- Diagnostic de généralisation et analyse de l'overfitting
- Explicabilité moderne (SHAP, Permutation Importance)
- Restitution via dashboard interactif et monitoring en production

---

## Structure du projet

```
RetentionInsight AI/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset brut Kaggle
│   └── processed_telco.csv                        # Dataset nettoyé et préparé
│
├── 1_eda.ipynb                 # Étape 1 : Analyse exploratoire descriptive (EDA)
├── 2_cleaning.py               # Étape 2 : Nettoyage, typage et optimisation RAM
├── 3_modeling.py               # Étape 3 : Pipeline d'entraînement, tuning, ROC & SHAP
├── 4_dashboard.py              # Étape 4 : Visualisation & Simulateur Streamlit
├── 5_monitoring.py             # Étape 5 : Suivi de dérive (Data & Model Drift)
│
├── model/
│   ├── churn_model.pkl         # Modèle champion final (Pipeline Scikit-Learn)
│   └── feature_names.json      # Schéma ordonné des variables d'entrée
├── plots/                      # Visualisations (ROC, SHAP, Seuil, Importances)
├── reports/                    # Rapports de monitoring générés (Evidently AI)
├── requirements.txt            # Dépendances Python
├── Dockerfile                  # Image Docker pour déploiement API
├── docker-compose.yml          # Orchestration Docker
├── RAPPORT.md                  # Rapport exécutif destiné à la direction
└── README.md                   # Ce fichier
```

---

## Pipeline de traitement

```
Données brutes (CSV)
        │
        ▼
[1_eda.ipynb]          ← Étape 1 : Analyse exploratoire & audit de qualité
        │
        ▼
[2_cleaning.py]        ← Étape 2 : Nettoyage, imputation & downcasting RAM
        │
        ▼  data/processed_telco.csv
        │
        ▼
[3_modeling.py]        ← Étape 3 : Ensembles, RandomizedSearchCV, seuil F1,
                                   courbes ROC superposées & analyse SHAP
        │
        ├──► model/churn_model.pkl
        └──► plots/ (roc_curves.png, shap_beeswarm.png, etc.)
        │
        ├─────────────────────────────┐
        ▼                             ▼
[4_dashboard.py]              [5_monitoring.py]
(Dashboard interactif Streamlit) (Suivi de performance & dérives)
```

---

## Résultats & Modélisation

### Comparaison des modèles (Validation Croisée 5-folds)

| Modèle | AUC-ROC (CV) | F1-Score | Précision | Rappel | Accuracy | Écart Train/Val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (majoritaire)** | 0.500 ± 0.000 | 0.000 | 0.000 | 0.000 | 0.735 | 0.00 % ✅ |
| **Arbre seul (profond)** | 0.656 ± 0.025 | 0.493 | 0.488 | 0.499 | 0.728 | 34.43 % ⚠️ |
| **Régression Logistique** | 0.846 ± 0.012 | 0.629 | 0.518 | 0.801 | 0.750 | 0.35 % ✅ |
| **Random Forest (régularisé)** | **0.847 ± 0.011** | **0.637** | **0.547** | **0.763** | **0.769** | **5.07 %** ✅ |
| **XGBoost (avec Early Stopping)** | **0.843 ± 0.011** | **0.630** | **0.538** | **0.761** | **0.763** | **8.86 %** ⚠️ |

### Évaluation finale du Champion sur le Jeu de Test (1 409 clients)
* **AUC-ROC Test** : **84,61 %** (AUC Train = 89,57 %, soit un écart de généralisation maîtrisé à **4,96 %**)
* **Rappel Churn** : **79,0 %** (près de 8 clients partants sur 10 détectés)
* **F1-Score Test** : **63,50 %**

---

## Installation

### Prérequis

- Python ≥ 3.9
- pip

### Environnement virtuel

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Via Docker

```bash
docker-compose up --build
```

---

## Utilisation

Exécuter les scripts dans l'ordre de la séquence :

```bash
# 1. Exploration descriptive (Notebook)
jupyter notebook 1_eda.ipynb

# 2. Nettoyage et préparation des données
python 2_cleaning.py

# 3. Modélisation complète, tuning & explicabilité
python 3_modeling.py

# 4. Lancement du simulateur et dashboard interactif
streamlit run 4_dashboard.py

# 5. Génération des rapports de monitoring (Evidently)
python 5_monitoring.py
```

---

## Bonnes pratiques & Standards d'ingénierie

Le projet applique les principes de développement logiciel appliqués à la Data Science :

1. **Prévention absolue du data leakage** : 
   - Aucune normalisation globale sur le jeu de données avant le split. 
   - Le `StandardScaler` est encapsulé dans un `ColumnTransformer` au sein d'un `Pipeline` Scikit-Learn ajusté uniquement sur les plis d'entraînement.

2. **Gestion défensive des données** :
   - Assertions systématiques de validation des schémas de colonnes en entrée.
   - Audit complet des valeurs manquantes (`NaN`) et imputation documentée.
   - Utilisation de copies explicites (`.copy()`) et proscription de `inplace=True`.

3. **Optimisation des ressources matérielles** :
   - Downcasting systématique des entiers et flottants (`float32`), réduisant l'empreinte mémoire de **96,5 %**.
   - Parallélisation multi-cœurs (`n_jobs=-1`) sur tous les estimateurs et recherches de grille.

4. **Reproductibilité** :
   - Fixation stricte des graines aléatoires (`random_state=42`) pour assurer des résultats déterministes.

---

## Données

**Source** : [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

* 7 043 clients
* 21 variables initiales (20 features + 1 cible `Churn`)
* Déséquilibre de classes : 26,5 % de churners

---

## Licence

Projet académique — L'École Multimédia · 2026
