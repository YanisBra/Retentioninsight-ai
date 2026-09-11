import importlib.util
import sys
import os
import pandas as pd
import numpy as np
import pytest

# Import dynamique de 2_cleaning.py car le nom commence par un chiffre
spec = importlib.util.spec_from_file_location(
    "cleaning", 
    os.path.join(os.path.dirname(__file__), "..", "2_cleaning.py")
)
cleaning = importlib.util.module_from_spec(spec)
sys.modules["cleaning"] = cleaning
spec.loader.exec_module(cleaning)

clean_total_charges = cleaning.clean_total_charges
validate_and_clean = cleaning.validate_and_clean


def test_clean_total_charges():
    # DataFrame fictif avec des espaces blancs et des valeurs normales
    df = pd.DataFrame({
        "TotalCharges": ["10.5", " ", "20.3", "30.0", "40.5"]
    })
    
    # Exécution
    df_cleaned = clean_total_charges(df)
    
    # Vérifications
    assert df_cleaned["TotalCharges"].isna().sum() == 0
    assert df_cleaned["TotalCharges"].dtype == "float32"
    # Médiane de [10.5, 20.3, 30.0, 40.5] = 25.15
    # L'espace blanc (" ") a dû être remplacé par 25.15
    assert df_cleaned.loc[1, "TotalCharges"] == pytest.approx(25.15)


def test_validate_and_clean():
    df = pd.DataFrame({
        "customerID": ["123-ABC", "456-DEF"],
        "SeniorCitizen": [0, 1],
        "tenure": [12, 24],
        "MonthlyCharges": [70.5, 80.0]
    })
    
    # Exécution
    df_cleaned = validate_and_clean(df)
    
    # Vérifications
    assert "customerID" not in df_cleaned.columns
    assert df_cleaned["MonthlyCharges"].dtype == "float32"
