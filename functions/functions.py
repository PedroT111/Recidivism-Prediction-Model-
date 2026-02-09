import streamlit as st
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from functions.translations import CATEGORY_LABELS, VARIABLE_LABELS

# functions to translate dataframes and variable labels for display purposes
def translate_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    df_disp = df.copy()
    for col, mapping in CATEGORY_LABELS.items():
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].replace(mapping)
    return df_disp

def var_label(col_name: str) -> str:
    return VARIABLE_LABELS.get(col_name, col_name)

def calculate_asociations(df, cols, target):
    resultados = []

    for col in cols:
        # Evitar columnas con muchos nulos o constantes
        if df[col].dropna().nunique() < 2:
            continue

        tabla = pd.crosstab(df[col], df[target])
        chi2, p, dof, exp = chi2_contingency(tabla)

        n = tabla.sum().sum()
        r, k = tabla.shape

        # Cramér's V corregido
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

        resultados.append({
            "Variable": col,
            "Chi²": round(chi2, 2),
            "p-valor": round(p, 5),
            "Cramér’s V": round(v, 3),
            "N categorías": tabla.shape[0],
        })

    if not resultados:
        return pd.DataFrame(columns=["Variable", "Chi²", "p-valor", "Cramér’s V", "N categorías"])

    asociaciones = (
        pd.DataFrame(resultados)
        .sort_values("Cramér’s V", ascending=False)
        .reset_index(drop=True)
    )
    return asociaciones


TARGET_LABELS_EN = {
    "Primario/a": "Primary",
    "Reincidente": "Recidivist",
    "Reiterante": "Repeat offender",
}

# function to display classification report in Streamlit
def display_classification_report(y_test, y_pred, target_encoder):
    """
    Render the classification report as a clean, lightly highlighted table in Streamlit.
    Highlights recall and F1-score for Recidivist and Repeat offender.
    """

    original_classes = target_encoder.classes_
    translated_classes = [TARGET_LABELS_EN.get(c, c) for c in original_classes]

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=translated_classes,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df = report_df.rename(index={
        "accuracy": "Accuracy",
        "macro avg": "Macro average",
        "weighted avg": "Weighted average",
    })

    cols_order = ["precision", "recall", "f1-score", "support"]
    report_df = report_df[cols_order]

    def highlight_key_cells(row):
        styles = [""] * len(row)
        if row.name in ["Recidivist", "Repeat offender"]:
            for i, col in enumerate(row.index):
                if col in ["recall", "f1-score"]:
                    styles[i] = "background-color: #4DA6FF; font-weight: 600;"  # naranja suave
        return styles

    styled = (
        report_df.style
        .format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1-score": "{:.3f}",
            "support": "{:.0f}",
        })
        .apply(highlight_key_cells, axis=1)
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "13px"), ("text-align", "center")]},
            {"selector": "td", "props": [("font-size", "12px")]},
        ])
    )
    st.write(styled)

