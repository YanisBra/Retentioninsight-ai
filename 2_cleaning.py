"""
2_cleaning.py — Script de nettoyage et préparation des données Telco

Projet  : RetentionInsight AI
Auteur  : Équipe Data Science

Étapes :
    1. Chargement du CSV brut
    2. Audit des NaN avant transformation
    3. Conversion de TotalCharges en float (errors='coerce') + imputation médiane
    4. Encodage one-hot des variables catégorielles (drop_first=True)
    5. StandardScaler sur tenure, MonthlyCharges, TotalCharges
    6. Optimisation RAM (dtypes)
    7. Sauvegarde dans data/processed_telco.csv
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 0. Constantes
# ─────────────────────────────────────────────
RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_PATH = "data/processed_telco.csv"

COLS_TO_SCALE = ["tenure", "MonthlyCharges", "TotalCharges"]
COL_TOTAL_CHARGES = "TotalCharges"
COL_CUSTOMER_ID = "customerID"

# Colonnes numériques connues (hors TotalCharges qui nécessite conversion)
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges"]


# ─────────────────────────────────────────────
# 1. Chargement
# ─────────────────────────────────────────────
def load_raw(path: str) -> pd.DataFrame:
    """
    Charge le CSV brut Telco et retourne une copie propre.

    Args:
        path (str): Chemin vers le fichier CSV brut.

    Returns:
        pd.DataFrame: DataFrame brut, copie indépendante.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        AssertionError: Si les colonnes obligatoires sont absentes.
    """
    df_raw = pd.read_csv(path)

    # Validation du schéma
    required_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    ]
    for col in required_cols:
        assert col in df_raw.columns, f"Colonne obligatoire manquante : '{col}'"

    print(f"[LOAD] {len(df_raw):,} lignes × {df_raw.shape[1]} colonnes chargées.")
    return df_raw.copy()


# ─────────────────────────────────────────────
# 2. Conversion & imputation de TotalCharges
# ─────────────────────────────────────────────
def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit TotalCharges en float via pd.to_numeric (errors='coerce'),
    puis impute les NaN résultants par la médiane.

    Args:
        df (pd.DataFrame): Dataset brut avec la colonne 'TotalCharges'.

    Returns:
        pd.DataFrame: Dataset avec TotalCharges en float32, sans NaN.

    Raises:
        AssertionError: Si 'TotalCharges' est absente.
    """
    assert COL_TOTAL_CHARGES in df.columns, (
        f"Colonne '{COL_TOTAL_CHARGES}' manquante."
    )

    df = df.copy()

    # NaN strategy: détecter AVANT conversion
    nan_before = df[COL_TOTAL_CHARGES].isna().sum()
    print(f"[NaN] TotalCharges — NaN avant conversion  : {nan_before}")

    # Conversion en numérique (les espaces blancs deviennent NaN)
    df[COL_TOTAL_CHARGES] = pd.to_numeric(
        df[COL_TOTAL_CHARGES], errors="coerce"
    )

    nan_after = df[COL_TOTAL_CHARGES].isna().sum()
    print(f"[NaN] TotalCharges — NaN après conversion  : {nan_after}")

    # NaN strategy: imputation par la médiane (robuste aux outliers)
    if nan_after > 0:
        median_val = df[COL_TOTAL_CHARGES].median()
        # Utiliser .loc[] pour éviter les copies intermédiaires
        df.loc[df[COL_TOTAL_CHARGES].isna(), COL_TOTAL_CHARGES] = median_val
        print(
            f"[IMPUTE] {nan_after} NaN imputés avec la médiane "
            f"({median_val:.2f}) dans '{COL_TOTAL_CHARGES}'."
        )

    # Vérification finale : aucun NaN résiduel
    assert df[COL_TOTAL_CHARGES].isna().sum() == 0, (
        "NaN résiduels dans TotalCharges après imputation."
    )

    # Optimisation RAM : float32 suffisant pour des montants monétaires
    df[COL_TOTAL_CHARGES] = df[COL_TOTAL_CHARGES].astype("float32")

    return df


# ─────────────────────────────────────────────
# 3. Nettoyage général & NaN audit
# ─────────────────────────────────────────────
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audite et nettoie les NaN restants.
    Supprime la colonne 'customerID' (non pertinente pour la modélisation).
    Optimise les dtypes RAM.

    Args:
        df (pd.DataFrame): Dataset après traitement de TotalCharges.

    Returns:
        pd.DataFrame: Dataset nettoyé avec dtypes optimisés.
    """
    df = df.copy()

    # Audit global des NaN
    nan_report = df.isna().sum()
    nan_cols = nan_report[nan_report > 0]
    if len(nan_cols) > 0:
        print(f"[NaN] Colonnes avec NaN restants :\n{nan_cols}")
    else:
        print("[NaN] Aucun NaN résiduel détecté — dataset propre.")

    # Suppression de customerID (identifiant non modélisable)
    if COL_CUSTOMER_ID in df.columns:
        df = df.drop(columns=[COL_CUSTOMER_ID])
        print(f"[DROP] Colonne '{COL_CUSTOMER_ID}' supprimée.")

    # Optimisation RAM — colonnes numériques entières
    for col in NUMERIC_COLS:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="integer")

    # Optimisation RAM — MonthlyCharges en float32
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"].astype("float32")

    return df


# ─────────────────────────────────────────────
# 4. Encodage one-hot des variables catégorielles
# ─────────────────────────────────────────────
def transform_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique pd.get_dummies(drop_first=True) sur toutes les variables
    catégorielles (dtype object ou category).

    Args:
        df (pd.DataFrame): Dataset nettoyé.

    Returns:
        pd.DataFrame: Dataset avec variables catégorielles encodées en 0/1.
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[ENCODE] Colonnes catégorielles détectées ({len(cat_cols)}) : {cat_cols}")

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convertir les colonnes booléennes issues de get_dummies en uint8
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype("uint8")

    print(f"[ENCODE] Shape après encodage : {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. Conversion type des colonnes numériques continues
# ─────────────────────────────────────────────
def transform_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes 'tenure', 'MonthlyCharges' et 'TotalCharges' en float32.
    Le scaling est délégué aux pipelines Scikit-Learn pour éviter les fuites de données.

    Args:
        df (pd.DataFrame): Dataset avec encodage one-hot appliqué.

    Returns:
        pd.DataFrame: Dataset avec colonnes converties en float32.

    Raises:
        AssertionError: Si une colonne est absente du DataFrame.
    """
    df = df.copy()

    for col in COLS_TO_SCALE:
        assert col in df.columns, (
            f"Colonne '{col}' manquante."
        )
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"La colonne '{col}' n'est pas numérique."
        )

    # Optimisation RAM : float32
    for col in COLS_TO_SCALE:
        df[col] = df[col].astype("float32")

    print(f"[CONVERT] Colonnes converties en float32 : {COLS_TO_SCALE}")
    return df


# ─────────────────────────────────────────────
# 6. Sauvegarde
# ─────────────────────────────────────────────
def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Sauvegarde le DataFrame final au format CSV.

    Args:
        df (pd.DataFrame): Dataset final prêt pour la modélisation.
        path (str): Chemin de sortie du CSV.
    """
    df.to_csv(path, index=False)
    print(f"[SAVE] Dataset sauvegardé → {path}")
    print(f"[SAVE] Shape finale : {df.shape}")


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────
def main() -> None:
    """
    Orchestre le pipeline de nettoyage complet des données.

    Étapes :
        1. load_raw          → df_raw
        2. clean_total_charges → df (TotalCharges propre)
        3. validate_and_clean  → df (types optimisés, NaN auditées)
        4. transform_categorical → df (one-hot encoding)
        5. transform_scale      → df_final (scaled)
        6. save_processed       → data/processed_telco.csv
    """
    print("=" * 60)
    print("  Pipeline de nettoyage — RetentionInsight AI")
    print("=" * 60)

    # Mémoire avant
    df_raw = load_raw(RAW_PATH)
    mem_before = df_raw.memory_usage(deep=True).sum() / 1024**2
    print(f"[RAM] Mémoire avant nettoyage : {mem_before:.2f} MB")

    # Étapes de transformation
    df = clean_total_charges(df_raw)
    df = validate_and_clean(df)
    df = transform_categorical(df)
    df_final = transform_scale(df)

    # Mémoire après
    mem_after = df_final.memory_usage(deep=True).sum() / 1024**2
    print(f"[RAM] Mémoire après nettoyage : {mem_after:.2f} MB")
    print(f"[RAM] Réduction : {(1 - mem_after / mem_before) * 100:.1f}%")

    # Audit final des NaN
    nan_final = df_final.isna().sum().sum()
    assert nan_final == 0, (
        f"ERREUR : {nan_final} NaN résiduels dans le dataset final !"
    )
    print(f"[CHECK] Zéro NaN dans le dataset final. ✓")

    # Sauvegarde
    save_processed(df_final, OUTPUT_PATH)

    print("=" * 60)
    print("  Pipeline de préparation des données terminé avec succès.")
    print("=" * 60)


if __name__ == "__main__":
    main()
