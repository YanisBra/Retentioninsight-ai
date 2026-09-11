"""
5_monitoring.py — Monitoring de la dérive des données et des performances

Projet  : RetentionInsight AI
Auteur  : Équipe Data Science

Compétence visée : D-06 — Piloter la performance de la solution d'IA via
des outils de monitoring.

Ce script génère deux rapports HTML auto-contenus dans reports/ :
    1. data_drift_report.html       : dérive statistique des features (Train→Test)
    2. model_performance_report.html: performances du modèle (ROC, matrice, F1)
    3. monitoring_summary.txt       : résumé textuel des métriques clés

Référence = jeu de Train (distribution connue à l'entraînement)
Courant   = jeu de Test  (nouvelles données simulant la production)
"""

# stdlib
import json
import os

# third-party
import base64
import io

import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

np.random.seed(42)

# ─────────────────────────────────────────────
# 0. Constantes
# ─────────────────────────────────────────────
DATA_PATH = "data/processed_telco.csv"
TARGET_COL = "Churn_Yes"
MODEL_PATH = "model/churn_model.pkl"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Chargement des données et du modèle
# ─────────────────────────────────────────────
def load_data_and_model() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object
]:
    """
    Charge le dataset, effectue le split Train/Test et charge le modèle.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, pipeline)
    """
    df = pd.read_csv(DATA_PATH)
    assert TARGET_COL in df.columns, f"Colonne cible '{TARGET_COL}' absente."

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipeline = joblib.load(MODEL_PATH)
    print(f"[LOAD] Dataset : {df.shape[0]:,} lignes | Modèle chargé depuis {MODEL_PATH}")
    return X_train, X_test, y_train, y_test, pipeline


# ─────────────────────────────────────────────
# Utilitaire : figure Matplotlib → base64 PNG
# ─────────────────────────────────────────────
def _fig_to_base64(fig: plt.Figure) -> str:
    """Encode une figure Matplotlib en chaîne base64 pour l'embarquer en HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _html_wrap(title: str, body_html: str) -> str:
    """Génère une page HTML complète auto-contenue."""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body{{font-family:"Segoe UI",Arial,sans-serif;background:#f5f7fa;
         color:#2d3436;margin:0;padding:0}}
    header{{background:#2d3436;color:#fff;padding:28px 40px}}
    header h1{{margin:0;font-size:1.6em}}
    header p{{margin:4px 0 0;opacity:.75;font-size:.9em}}
    main{{max-width:1100px;margin:32px auto;padding:0 24px}}
    section{{background:#fff;border-radius:10px;padding:28px 32px;
             margin-bottom:28px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
    h2{{color:#2d3436;border-bottom:2px solid #dfe6e9;padding-bottom:8px}}
    table{{border-collapse:collapse;width:100%;font-size:.9em}}
    th{{background:#2d3436;color:#fff;padding:9px 14px;text-align:left}}
    td{{padding:8px 14px;border-bottom:1px solid #dfe6e9}}
    tr:nth-child(even){{background:#f8f9fa}}
    .badge{{display:inline-block;padding:3px 10px;border-radius:12px;
            font-weight:600;font-size:.82em}}
    .ok{{background:#d4efdf;color:#1e8449}}
    .warn{{background:#fdebd0;color:#d35400}}
    img{{max-width:100%;border-radius:8px;margin-top:12px}}
    footer{{text-align:center;padding:20px;opacity:.5;font-size:.8em}}
  </style>
</head>
<body>
<header>
  <h1>RetentionInsight AI — {title}</h1>
  <p>Projet #3 · L'École Multimédia · Généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</header>
<main>{body_html}</main>
<footer>RetentionInsight AI · Monitoring Report</footer>
</body></html>"""


# ─────────────────────────────────────────────
# 2. Rapport de dérive des données
# ─────────────────────────────────────────────
def run_data_drift_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Génère un rapport HTML de dérive des données (Train → Test).
    Compare les distributions des features entre le jeu de référence
    (Train) et le jeu courant (Test / production simulée).
    Utilise des statistiques descriptives et un graphique des écarts de moyenne.

    Args:
        X_train (pd.DataFrame): Données de référence (entraînement).
        X_test (pd.DataFrame): Données courantes (test / production).
    """
    # Calcul des écarts de distribution
    drift = pd.DataFrame({
        "mean_train": X_train.mean(),
        "mean_test": X_test.mean(),
        "std_train": X_train.std(),
        "std_test": X_test.std(),
    })
    drift["delta_mean"] = (drift["mean_test"] - drift["mean_train"]).abs()
    drift["drift_pct"] = (
        drift["delta_mean"] / drift["mean_train"].abs().clip(lower=1e-6) * 100
    ).round(1)
    drift = drift.sort_values("delta_mean", ascending=False)
    top10 = drift.head(10)

    # Graphique barh des Top 10 dérives
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if v > 5 else "#3498db" for v in top10["drift_pct"]]
    bars = ax.barh(top10.index[::-1], top10["delta_mean"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.65)
    for bar, val in zip(bars, top10["delta_mean"][::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_title("Top 10 Features — Écart de moyenne (Train vs Test)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Δ moyenne absolue")
    sns.despine()
    plt.tight_layout()
    drift_img = _fig_to_base64(fig)
    plt.close()

    # Tableau HTML
    rows = ""
    for feat, row in top10.iterrows():
        badge = ("<span class='badge warn'>Dérive</span>"
                 if row["drift_pct"] > 5
                 else "<span class='badge ok'>Stable</span>")
        rows += (f"<tr><td>{feat}</td><td>{row['mean_train']:.4f}</td>"
                 f"<td>{row['mean_test']:.4f}</td><td>{row['delta_mean']:.4f}</td>"
                 f"<td>{row['drift_pct']:.1f}%</td><td>{badge}</td></tr>")

    body = f"""
    <section>
      <h2>📊 Dérive des données — Top 10 features</h2>
      <p>Référence = Train ({X_train.shape[0]:,} obs) · Courant = Test ({X_test.shape[0]:,} obs)</p>
      <img src="data:image/png;base64,{drift_img}" alt="data drift">
      <table>
        <tr><th>Feature</th><th>Moy. Train</th><th>Moy. Test</th>
            <th>Δ absolu</th><th>Dérive %</th><th>Statut</th></tr>
        {rows}
      </table>
    </section>"""

    save_path = os.path.join(REPORTS_DIR, "data_drift_report.html")
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(_html_wrap("Rapport de Dérive des Données", body))
    print(f"[REPORT] Rapport de dérive → {save_path}")


# ─────────────────────────────────────────────
# 3. Rapport de performance du modèle
# ─────────────────────────────────────────────
def run_classification_performance_report(
    pipeline: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Génère un rapport HTML de performance de classification.
    Inclut : courbe ROC, matrice de confusion, classification report,
    et comparaison AUC Train/Test.

    Args:
        pipeline: Pipeline scikit-learn entraîné.
        X_train, X_test (pd.DataFrame): Features train/test.
        y_train, y_test (pd.Series): Cibles train/test.
    """
    y_pred = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]

    auc_train = roc_auc_score(y_train, y_proba_train)
    auc_test = roc_auc_score(y_test, y_proba_test)
    delta = abs(auc_train - auc_test)

    # ── Figure 1 : Courbe ROC ──
    fig_roc, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba_test)
    ax.plot(fpr, tpr, color="#2ecc71", linewidth=2.5,
            label=f"Random Forest v2 (AUC={auc_test:.3f})")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1.2, label="Aléatoire (0.500)")
    ax.set_xlabel("Taux de Faux Positifs")
    ax.set_ylabel("Taux de Vrais Positifs")
    ax.set_title("Courbe ROC — Test Set", fontweight="bold")
    ax.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()
    roc_img = _fig_to_base64(fig_roc)
    plt.close()

    # ── Figure 2 : Matrice de confusion ──
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Non-Churn", "Churn"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Matrice de Confusion", fontweight="bold")
    plt.tight_layout()
    cm_img = _fig_to_base64(fig_cm)
    plt.close()

    # ── Classification report → tableau HTML ──
    from sklearn.metrics import precision_score, recall_score
    cr_dict = {
        "Non-Churn (0)": {
            "precision": precision_score(y_test, y_pred, pos_label=0),
            "recall": recall_score(y_test, y_pred, pos_label=0),
            "f1": f1_score(y_test, y_pred, pos_label=0),
        },
        "Churn (1)": {
            "precision": precision_score(y_test, y_pred, pos_label=1),
            "recall": recall_score(y_test, y_pred, pos_label=1),
            "f1": f1_score(y_test, y_pred, pos_label=1),
        },
    }
    cr_rows = "".join(
        f"<tr><td>{cls}</td><td>{m['precision']:.3f}</td>"
        f"<td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td></tr>"
        for cls, m in cr_dict.items()
    )

    ovf_badge = (
        "<span class='badge warn'>Overfitting résiduel</span>"
        if delta > 0.05
        else "<span class='badge ok'>Généralisation correcte</span>"
    )

    body = f"""
    <section>
      <h2>🎯 Indicateurs de performance — Test Set</h2>
      <table>
        <tr><th>Métrique</th><th>Valeur</th><th>Statut</th></tr>
        <tr><td>AUC-ROC Train</td><td>{auc_train:.4f}</td><td>—</td></tr>
        <tr><td>AUC-ROC Test</td><td>{auc_test:.4f}</td>
            <td><span class='badge ok'>Bon</span></td></tr>
        <tr><td>|ΔAUC| Train–Test</td><td>{delta:.4f}</td><td>{ovf_badge}</td></tr>
        <tr><td>Accuracy Test</td>
            <td>{(y_pred == y_test.values).mean():.4f}</td><td>—</td></tr>
        <tr><td>F1 macro</td><td>{f1_score(y_test, y_pred, average='macro'):.4f}</td><td>—</td></tr>
      </table>
    </section>
    <section>
      <h2>📈 Courbe ROC &amp; Matrice de Confusion</h2>
      <div style="display:flex;gap:24px;flex-wrap:wrap">
        <img src="data:image/png;base64,{roc_img}" alt="ROC" style="max-width:420px">
        <img src="data:image/png;base64,{cm_img}" alt="Confusion Matrix" style="max-width:360px">
      </div>
    </section>
    <section>
      <h2>📋 Rapport de Classification</h2>
      <table>
        <tr><th>Classe</th><th>Précision</th><th>Rappel</th><th>F1-Score</th></tr>
        {cr_rows}
      </table>
    </section>"""

    save_path = os.path.join(REPORTS_DIR, "model_performance_report.html")
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(_html_wrap("Rapport de Performance du Modèle", body))
    print(f"[REPORT] Rapport de performance → {save_path}")


# ─────────────────────────────────────────────
# 4. Rapport de synthèse textuel (fallback)
# ─────────────────────────────────────────────
def run_text_summary(
    pipeline: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """
    Génère un résumé de monitoring textuel (fallback si Evidently indisponible).
    Calcule les statistiques de dérive et les performances et les sauvegarde
    dans reports/monitoring_summary.txt.

    Args:
        pipeline: Pipeline scikit-learn entraîné.
        X_train, X_test (pd.DataFrame): Features train/test.
        y_train, y_test (pd.Series): Cibles train/test.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, classification_report
    )

    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]

    auc_train = roc_auc_score(y_train, y_proba_train)
    auc_test = roc_auc_score(y_test, y_proba_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    # Dérive des moyennes des features
    drift_summary = (X_test.mean() - X_train.mean()).abs().sort_values(ascending=False)
    top_drift = drift_summary.head(5)

    lines = [
        "=" * 60,
        "  MONITORING SUMMARY — RetentionInsight AI",
        "=" * 60,
        f"  Date génération : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "  [PERFORMANCE]",
        f"  AUC-ROC Train  : {auc_train:.4f}",
        f"  AUC-ROC Test   : {auc_test:.4f}",
        f"  |ΔAUC|         : {abs(auc_train - auc_test):.4f}",
        f"  Accuracy Test  : {acc:.4f}",
        f"  F1-Score Test  : {f1:.4f}",
        "",
        "  [DATA DRIFT — Top 5 features avec le plus grand écart de moyenne]",
    ]
    for feat, val in top_drift.items():
        lines.append(f"  {feat:<40} Δmoy={val:.4f}")

    lines += [
        "",
        "  [CLASSIFICATION REPORT]",
        classification_report(y_test, y_pred_test),
        "=" * 60,
    ]

    summary_text = "\n".join(lines)
    print(summary_text)

    save_path = os.path.join(REPORTS_DIR, "monitoring_summary.txt")
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text)
    print(f"\n[SAVE] Résumé de monitoring → {save_path}")


# ─────────────────────────────────────────────
# 5. Pipeline principal
# ─────────────────────────────────────────────
def main() -> None:
    """
    Orchestre le monitoring complet du modèle.

    Étapes :
        1. Chargement des données + modèle.
        2. Rapport de dérive des données (HTML auto-contenu).
        3. Rapport de performance du modèle (HTML auto-contenu).
        4. Résumé textuel de monitoring (TXT).
    """
    print("=" * 60)
    print("  Monitoring de Production — RetentionInsight AI")
    print("=" * 60)

    X_train, X_test, y_train, y_test, pipeline = load_data_and_model()

    run_data_drift_report(X_train, X_test)
    run_classification_performance_report(pipeline, X_train, X_test, y_train, y_test)
    run_text_summary(pipeline, X_train, X_test, y_train, y_test)

    print("\n  Monitoring terminé. Rapports dans reports/")


if __name__ == "__main__":
    main()
