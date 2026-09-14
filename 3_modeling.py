"""
3_modeling.py — Pipeline de modélisation et évaluation Machine Learning

Ce script orchestre l'ensemble du cycle de modélisation du projet :
    1. Chargement du dataset nettoyé et split stratifié 80/20.
    2. Définition du ColumnTransformer pour le scaling sans fuite.
    3. Tuning dynamique par RandomizedSearchCV (cv=5) sur Random Forest.
    4. Évaluation comparée par Validation Croisée (cv=5) :
       - Dummy Classifier (Baseline naïve)
       - Régression Logistique
       - Arbre de Décision seul (profond)
       - Random Forest régularisé
       - XGBoost (avec scale_pos_weight & Early Stopping)
    5. Recherche du seuil de décision optimal (F1-score) via cross_val_predict.
    6. Analyse d'importance des variables (Gini vs. Permutation).
    7. Tracé des courbes ROC superposées (figure reine).
    8. Explicabilité avancée par SHAP (Beeswarm global & Waterfall local).
    9. Évaluation et sauvegarde du modèle champion final.
"""

# stdlib
import json
import os

# third-party
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Reproductibilité globale
np.random.seed(42)

# Constantes
DATA_PATH = "data/processed_telco.csv"
TARGET_COL = "Churn_Yes"
RANDOM_STATE = 42
MODEL_DIR = "model"
PLOTS_DIR = "plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_and_split(
    path: str,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Charge le dataset pré-traité et effectue un split stratifié 80/20.

    Args:
        path (str): Chemin vers le CSV nettoyé.
        target (str): Nom de la colonne cible.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(path)
    assert target in df.columns, f"Colonne cible '{target}' absente."
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[SPLIT] Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def get_preprocessor() -> ColumnTransformer:
    """
    Construit le ColumnTransformer pour normaliser les variables continues.

    Returns:
        ColumnTransformer: Le transformateur configuré.
    """
    return ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), ["tenure", "MonthlyCharges", "TotalCharges"])
        ],
        remainder="passthrough",
    )


def evaluer(
    model: Pipeline,
    nom: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: StratifiedKFold,
) -> dict:
    """
    Évalue un pipeline par validation croisée multi-métriques (cv=5).

    Args:
        model (Pipeline): Pipeline Scikit-Learn.
        nom (str): Nom d'affichage du modèle.
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Cible d'entraînement.
        cv_strategy (StratifiedKFold): Stratégie de CV.

    Returns:
        dict: Métriques moyennes de validation croisée.
    """
    res = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring=["roc_auc", "f1", "precision", "recall", "accuracy"],
        return_train_score=True,
        n_jobs=-1,
    )

    train_roc = res["train_roc_auc"].mean()
    val_roc = res["test_roc_auc"].mean()
    diff = abs(train_roc - val_roc)

    warning_str = " ⚠️" if diff > 0.05 else " ✅"

    print(
        f"  [CV] {nom:<30} | AUC Val : {val_roc:.4f} | "
        f"Ecart Train/Val : {diff:.2%}{warning_str}"
    )

    return {
        "modèle": nom,
        "AUC": f"{val_roc:.4f} ± {res['test_roc_auc'].std():.3f}",
        "F1": f"{res['test_f1'].mean():.4f}",
        "précision": f"{res['test_precision'].mean():.4f}",
        "rappel": f"{res['test_recall'].mean():.4f}",
        "accuracy": f"{res['test_accuracy'].mean():.4f}",
        "overfit_gap": f"{diff:.4f}",
    }


def optimize_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_strategy: StratifiedKFold,
) -> float:
    """
    Optimise le seuil de classification en maximisant le F1-score.

    Args:
        pipeline (Pipeline): Le pipeline du modèle.
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Cible d'entraînement.
        cv_strategy (StratifiedKFold): Stratégie de validation croisée.

    Returns:
        float: Le seuil de décision optimal.
    """
    print("\n" + "─" * 58)
    print("  [SEUIL] Optimisation du seuil de décision (F1-score)")
    print("─" * 58)

    y_proba_cv = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv_strategy,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    seuils = np.arange(0.10, 0.90, 0.02)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for s in seuils:
        y_pred = (y_proba_cv >= s).astype(int)
        f1_scores.append(f1_score(y_train, y_pred))
        precision_scores.append(precision_score(y_train, y_pred))
        recall_scores.append(recall_score(y_train, y_pred))

    best_idx = np.argmax(f1_scores)
    best_seuil = seuils[best_idx]

    print(f"  🏆 Seuil optimal trouvé : {best_seuil:.2f}")
    print(f"  │  F1-Score CV : {f1_scores[best_idx]:.4f}")
    print(f"  │  Précision   : {precision_scores[best_idx]:.4f}")
    print(f"  │  Rappel       : {recall_scores[best_idx]:.4f}")

    # Plot & Save
    plt.figure(figsize=(7, 4.5))
    plt.plot(seuils, f1_scores, label="F1-Score", color="firebrick", lw=2)
    plt.plot(
        seuils, precision_scores, label="Précision", color="darkgreen", linestyle="--"
    )
    plt.plot(
        seuils, recall_scores, label="Rappel", color="royalblue", linestyle="--"
    )
    plt.axvline(
        best_seuil, color="black", linestyle=":", label=f"Seuil Optimal = {best_seuil:.2f}"
    )
    plt.title("Optimisation du seuil de décision (Random Forest)")
    plt.xlabel("Seuil")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "threshold_optimization.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  └─ Graphique sauvegardé → {plot_path}")

    return float(best_seuil)


def analyse_importances(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """
    Calcule et compare l'importance Gini et l'importance par permutation.

    Args:
        pipeline (Pipeline): Pipeline Random Forest entraîné.
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Cible d'entraînement.
    """
    print("\n" + "─" * 58)
    print("  [IMPORTANCES] Comparaison Gini vs. Permutation")
    print("─" * 58)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
    )

    pipeline.fit(X_tr, y_tr)

    rf_model = pipeline.steps[-1][1]
    preprocessor = pipeline.steps[0][1]
    feature_names = preprocessor.get_feature_names_out()
    cleaned_names = [
        f.replace("scaler__", "").replace("remainder__", "")
        for f in feature_names
    ]

    gini_imp = pd.Series(rf_model.feature_importances_, index=cleaned_names)
    top_gini = gini_imp.sort_values(ascending=False).head(10)

    # Permutation importance
    perm = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    perm_imp = pd.Series(perm.importances_mean, index=X_val.columns)
    top_perm = perm_imp.sort_values(ascending=False).head(10)

    print("  🏆 Top 5 Gini (Entraînement) :")
    for rank, (feat, val) in enumerate(top_gini.head(5).items(), 1):
        print(f"    #{rank} {feat:<30} : {val:.4f}")

    print("\n  🏆 Top 5 Permutation (Validation hors-pli) :")
    for rank, (feat, val) in enumerate(top_perm.head(5).items(), 1):
        print(f"    #{rank} {feat:<30} : {val:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    top_gini.sort_values(ascending=True).plot.barh(ax=axes[0], color="firebrick")
    axes[0].set_title("Importance par Impureté (Gini)")
    axes[0].set_xlabel("Importance")

    top_perm.sort_values(ascending=True).plot.barh(ax=axes[1], color="royalblue")
    axes[1].set_title("Importance par Permutation (Validation)")
    axes[1].set_xlabel("Chute moyenne d'AUC-ROC")

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "permutation_importance.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  └─ Graphique comparatif sauvegardé → {plot_path}")


def plot_roc_curves_comparison(
    finalistes: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Trace et sauvegarde les courbes ROC superposées des modèles sur le test set.

    Args:
        finalistes (dict): Modèles à comparer.
        X_train (pd.DataFrame): Features train.
        y_train (pd.Series): Cible train.
        X_test (pd.DataFrame): Features test.
        y_test (pd.Series): Cible test.
    """
    print("\n" + "─" * 58)
    print("  [ROC] Tracé des Courbes ROC Superposées (Figure Reine)")
    print("─" * 58)
    fig, ax = plt.subplots(figsize=(8, 6))

    for nom, modele in finalistes.items():
        modele.fit(X_train, y_train)
        RocCurveDisplay.from_estimator(modele, X_test, y_test, name=nom, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Aléatoire (AUC = 0.50)")
    ax.set_title(
        "Comparaison des Courbes ROC (Jeu de Test)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)")
    ax.set_ylabel("Taux de Vrais Positifs (Rappel)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  🏆 Courbes ROC sauvegardées → {plot_path}")


def explain_with_shap(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Génère les visualisations SHAP (Beeswarm global & Waterfall local).

    Args:
        pipeline (Pipeline): Modèle champion entraîné.
        X_train (pd.DataFrame): Train set.
        y_train (pd.Series): Train target.
        X_test (pd.DataFrame): Test set.
        y_test (pd.Series): Test target.
    """
    print("\n" + "─" * 58)
    print("  [SHAP] Explicabilité Moderne (TreeExplainer)")
    print("─" * 58)

    pipeline.fit(X_train, y_train)
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_test_t = preprocessor.transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()

    feature_names = preprocessor.get_feature_names_out()
    cleaned_names = [
        f.replace("scaler__", "").replace("remainder__", "")
        for f in feature_names
    ]

    print("  [SHAP] Calcul des valeurs SHAP avec TreeExplainer...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_t)
    shap_values.feature_names = cleaned_names

    # Si classifieur binaire multi-classe (ex: Random Forest), sélectionner la classe 1
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values_churn = shap_values[:, :, 1]
    else:
        shap_values_churn = shap_values

    # 1. Beeswarm plot global
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values_churn, max_display=12, show=False)
    plt.title(
        "Vue Globale SHAP — Impact des variables sur le Churn",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    beeswarm_path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  🏆 Beeswarm SHAP sauvegardé → {beeswarm_path}")

    # 2. Waterfall plot local pour un vrai churner
    churners_idx = np.where(y_test.values == 1)[0]
    idx_target = int(churners_idx[8]) if len(churners_idx) > 8 else int(churners_idx[0])

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values_churn[idx_target], max_display=10, show=False)
    plt.title(
        f"Vue Locale SHAP — Client n°{idx_target} (Profil Churner)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    waterfall_path = os.path.join(PLOTS_DIR, "shap_waterfall.png")
    plt.savefig(waterfall_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  🏆 Waterfall SHAP sauvegardé → {waterfall_path}")


def evaluate_final_champion(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    nom: str,
) -> None:
    """Entraîne et affiche les performances finales du modèle champion sur le test set."""
    print("\n" + "═" * 58)
    print(f"  🏆  ÉVALUATION FINALE DU CHAMPION : {nom}")
    print("═" * 58)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy Test : {acc:.4f}")
    print(f"  F1-Score Test : {f1:.4f}")
    print(f"  ROC-AUC Test  : {roc:.4f}")
    print(f"\n  Matrice de confusion :")
    print(f"    [VN: {cm[0][0]} | FP: {cm[0][1]}]")
    print(f"    [FN: {cm[1][0]} | VP: {cm[1][1]}]")
    print("\n  Classification Report :")
    for line in classification_report(y_test, y_pred).split("\n"):
        print(f"    {line}")
    print("═" * 58)


def main() -> None:
    """Cycle de vie complet du modèle."""
    print("=" * 58)
    print("  Pipeline d'Apprentissage & Modélisation Consolidé")
    print("=" * 58)

    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH, TARGET_COL)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = get_preprocessor()

    # 1. Optimisation par RandomizedSearchCV pour le Random Forest (cours ch. 5 & 8)
    print("\n" + "─" * 58)
    print("  [TUNING] Recherche d'hyperparamètres Random Forest (RandomizedSearchCV)")
    print("─" * 58)

    rf_base_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    rf_param_dist = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, 15, 20],
        "classifier__min_samples_split": [5, 10, 20],
        "classifier__min_samples_leaf": [3, 5, 10],
    }

    rs = RandomizedSearchCV(
        estimator=rf_base_pipeline,
        param_distributions=rf_param_dist,
        n_iter=30,  # 30 combinaisons aléatoires
        cv=cv_strategy,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    print("  [CV] Exécution de RandomizedSearchCV (30 itérations, cv=5)...")
    rs.fit(X_train, y_train)
    best_params = rs.best_params_
    print(f"  🏆 Meilleurs hyperparamètres trouvés : {best_params}")
    print(f"  │  Meilleur score ROC-AUC (CV) : {rs.best_score_:.4f}")

    cleaned_params = {
        k.replace("classifier__", ""): v for k, v in best_params.items()
    }
    cleaned_params["best_auc_cv"] = round(float(rs.best_score_), 6)
    best_params_path = "best_params_v2.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_params, f, indent=2)
    print(f"  └─ Paramètres sauvegardés → {best_params_path}")

    # 2. Définition des pipelines comparatifs (tableau complet ch. 8.2)
    pipelines = {
        "Dummy (majoritaire)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Régression Logistique": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                )),
            ]
        ),
        "Arbre seul (profond)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(
                    max_depth=None,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "Random Forest (régularisé)": rs.best_estimator_,
        "XGBoost (standard + bal.)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=2.77,
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]
        ),
    }

    # 3. Cross-Validation
    print("\n" + "─" * 58)
    print("  [EVAL] Validation Croisée Stratifiée (cv=5) — Tableau Comparatif")
    print("─" * 58)
    resultats = []
    for nom, pipe in pipelines.items():
        res = evaluer(pipe, nom, X_train, y_train, cv_strategy)
        resultats.append(res)

    print("\n" + pd.DataFrame(resultats).to_string(index=False))

    # 4. XGBoost avec Early Stopping
    print("\n" + "─" * 58)
    print("  [ES] XGBoost avec Early Stopping")
    print("─" * 58)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
    )

    prep_fit = get_preprocessor().fit(X_tr)
    X_tr_t = prep_fit.transform(X_tr)
    X_val_t = prep_fit.transform(X_val)

    xgb_es = XGBClassifier(
        n_estimators=2000,
        early_stopping_rounds=50,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.77,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_es.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)], verbose=False)
    best_iter = xgb_es.best_iteration
    print(f"  🏆 Itération optimale d'arbres trouvée : {best_iter}")

    xgb_final = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=best_iter,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=2.77,
                eval_metric="auc",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )
    pipelines["XGBoost (Early Stopped)"] = xgb_final

    # 5. Optimisation du seuil de décision & Analyse d'importance
    rf_pipeline = pipelines["Random Forest (régularisé)"]
    best_seuil = optimize_threshold(rf_pipeline, X_train, y_train, cv_strategy)
    analyse_importances(rf_pipeline, X_train, y_train)

    # 6. Courbes ROC superposées (Figure Reine)
    finalistes = {
        "Rég. Logistique": pipelines["Régression Logistique"],
        "Random Forest (régularisé)": rf_pipeline,
        "XGBoost (Early Stopped)": xgb_final,
    }
    plot_roc_curves_comparison(finalistes, X_train, y_train, X_test, y_test)

    # 7. Explicabilité SHAP
    explain_with_shap(rf_pipeline, X_train, y_train, X_test, y_test)

    # 8. Évaluation finale du champion (Random Forest v2)
    evaluate_final_champion(
        rf_pipeline, X_train, y_train, X_test, y_test, "Random Forest (régularisé)"
    )

    # 9. Sauvegarde du modèle champion et des métadonnées
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    joblib.dump(rf_pipeline, model_path)
    print(f"\n[SAVE] Modèle champion sauvegardé → {model_path}")

    feature_names_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"[SAVE] Noms de features sauvegardés → {feature_names_path}")


if __name__ == "__main__":
    main()
