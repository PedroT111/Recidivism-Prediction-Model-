# model.py
from typing import Any, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from data.data_loader import load_data
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

def prepare_multiclass_df(df: pd.DataFrame):
    """
    Prepare the dataset for the multiclass model.
    Classes: Primario/a, Reincidente, Reiterante.
    """
    df_model = df.copy()

    # Adjust target labels
    df_model["es_reincidente_descripcion"] = df_model["es_reincidente_descripcion"].replace(
        {
            "Primario/a": "Primario/a",
            "Reincidente (art. 50 CP)": "Reincidente",
            "Reiterante": "Reiterante",
        }
    )

    target = "es_reincidente_descripcion"
    num_cols = ["edad", "duracion_condena_anios"]
    bin_cols = ["participa_programa_pre_libertad", "participacion_actividades_deportivas"]
    cat_cols = [
        "delito1_descripcion",
        "nivel_instruccion_descripcion",
        "ultima_situacion_laboral_descripcion",
        "calificacion_conducta_descripcion",
        "ultima_provincia_residencia_descripcion",
        "tiene_periodo_progresividad_descripcion",
        "tipo_infraccion_disciplinaria_descripcion",
    ]

    # Encode target
    target_encoder = LabelEncoder()
    df_model[target] = target_encoder.fit_transform(df_model[target])

    # Encode categorical features
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    X = df_model[num_cols + bin_cols + cat_cols]
    y = df_model[target]

    return X, y, num_cols, bin_cols, cat_cols, target_encoder, label_encoders

# Internal helper function
def _train_multiclass_random_forest(
    test_size: float,
    random_state: int,
    rf_params: Dict[str, Any],
) -> Dict[str, Any]:
    df = load_data()
    (
        X,
        y,
        num_cols,
        bin_cols,
        cat_cols,
        target_encoder,
        label_encoders,
    ) = prepare_multiclass_df(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        **rf_params,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    class_names = list(target_encoder.classes_)

    # numbers
    labels_int = list(range(len(class_names)))

    # global metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # confusion matrix 
    cm = confusion_matrix(y_test, y_pred, labels=labels_int)

    # metrics by class
    recall_vals = recall_score(y_test, y_pred, labels=labels_int, average=None, zero_division=0)
    precision_vals = precision_score(y_test, y_pred, labels=labels_int, average=None, zero_division=0)

    recall_by_class = dict(zip(class_names, recall_vals))
    precision_by_class = dict(zip(class_names, precision_vals))

    # Report
    report_str = classification_report(
        y_test,
        y_pred,
        labels=labels_int,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels_int,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Importances
    feature_names = num_cols + bin_cols + cat_cols
    importances = (
        pd.DataFrame(
            {"feature": feature_names, "importance": rf.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "model": rf,
        "cm": cm,
        "importances": importances,
        "target_encoder": target_encoder,
        "class_names": class_names,

        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "recall_by_class": recall_by_class,
        "precision_by_class": precision_by_class,

        "report_str": report_str,
        "report_dict": report_dict,
    }

# train baseline model
def train_multiclass_random_forest_baseline(
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a **baseline** multiclass Random Forest model with conservative,
    non-optimized hyperparameters, to serve as a performance reference.
    """
    rf_params = dict(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight=None,
    )
    return _train_multiclass_random_forest(
        test_size=test_size,
        random_state=random_state,
        rf_params=rf_params,
    )

# train optimized model
def train_multiclass_random_forest_optimized(
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train an **optimized** multiclass Random Forest model, using
    tuned hyperparameters that account for class imbalance and
    regularization.
    """
    rf_params = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=3,
        class_weight="balanced",
    )
    return _train_multiclass_random_forest(
        test_size=test_size,
        random_state=random_state,
        rf_params=rf_params,
    )

def prepare_binary_df(df: pd.DataFrame):
    """
    Prepare the dataset for a binary model:
    Primario/a vs Reincidente (Reincidente + Reiterante).
    """
    df_binary = df.copy()
    df_binary["es_reincidente_binaria"] = df_binary["es_reincidente_descripcion"].replace(
        {
            "Primario/a": "Primario/a",
            "Reincidente (art. 50 CP)": "Reincidente",
            "Reiterante": "Reincidente",
        }
    )

    target = "es_reincidente_binaria"
    num_cols = ["edad", "duracion_condena_anios"]
    bin_cols = ["participa_programa_pre_libertad", "participacion_actividades_deportivas"]
    cat_cols = [
        "delito1_descripcion",
        "nivel_instruccion_descripcion",
        "ultima_situacion_laboral_descripcion",
        "calificacion_conducta_descripcion",
        "ultima_provincia_residencia_descripcion",
        "tiene_periodo_progresividad_descripcion",
        "tipo_infraccion_disciplinaria_descripcion",
    ]

    target_encoder = LabelEncoder()
    df_binary[target] = target_encoder.fit_transform(df_binary[target])

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_binary[col] = le.fit_transform(df_binary[col].astype(str))
        label_encoders[col] = le

    X = df_binary[num_cols + bin_cols + cat_cols]
    y = df_binary[target]

    return (
        df_binary,
        X,
        y,
        num_cols,
        bin_cols,
        cat_cols,
        target_encoder,
        label_encoders,
        target,
    )


def train_balanced_rf_binary(test_size: float = 0.3, random_state: int = 42):
    """
    Train a binary Balanced Random Forest and return:
    - model
    - metrics
    - predicted probabilities (for interactive demo)
    """
    df = load_data()
    (
    df_binary,
    X,
    y,
    num_cols,
    bin_cols,
    cat_cols,
    target_encoder,
    label_encoders,
    target_name,
) = prepare_binary_df(df)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    brf = BalancedRandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=3,
        sampling_strategy="not minority",  # partial rebalancing
        random_state=random_state,
        n_jobs=-1,
    )
    brf.fit(X_train, y_train)

    y_pred = brf.predict(X_test)
    y_proba = brf.predict_proba(X_test)

    class_names = list(target_encoder.classes_)       
    labels_int = list(range(len(class_names)))              

    if "Reincidente" in class_names:
        pos_label = int(class_names.index("Reincidente"))
    else:
        pos_label = 1 if len(class_names) > 1 else 0

    # global metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # metrics by class
    precision_pos = precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    recall_pos = recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)
    f1_pos = f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)

    # report
    report_str = classification_report(
        y_test,
        y_pred,
        labels=labels_int,
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels_int,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels_int)

    # --- importances ---
    feature_names = num_cols + bin_cols + cat_cols
    importances = (
        pd.DataFrame(
            {"feature": feature_names, "importance": brf.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,

        "model": brf,
        "cm": cm,
        "importances": importances,
        "target_encoder": target_encoder,
        "class_names": class_names,

        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "pos_label": pos_label,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "report_str": report_str,
        "report_dict": report_dict,
    }


def preprocess_binary_input(input_dict: dict, artifacts: dict):
    """
    Preprocess input data for the binary model interactive demo.
    """
    num_cols = artifacts["num_cols"]
    bin_cols = artifacts["bin_cols"]
    cat_cols = artifacts["cat_cols"]
    label_encoders = artifacts["label_encoders"]

    df_new = pd.DataFrame([input_dict])

    # Use the same LabelEncoders for categorical features
    for col in cat_cols:
        le = label_encoders[col]
        value = str(df_new[col].iloc[0])
        # If a new category appears, add it to the encoder
        if value not in le.classes_:
            le.classes_ = list(le.classes_) + [value]
        df_new[col] = le.transform(df_new[col].astype(str))

    X_new = df_new[num_cols + bin_cols + cat_cols]
    return X_new
