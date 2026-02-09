import streamlit as st
import matplotlib.pyplot as plt
from functions.chart_utils import plot_confusion_matrix, plot_confusion_matrix_normalized
from functions.functions import TARGET_LABELS_EN, display_classification_report
from model import (
    train_multiclass_random_forest_baseline,
    train_balanced_rf_binary,
    train_multiclass_random_forest_optimized,
)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

@st.cache_resource
def get_base_results():
    return train_multiclass_random_forest_baseline()

@st.cache_resource
def get_multiclase_results():
    return train_multiclass_random_forest_optimized()

@st.cache_resource
def get_binario_results():
    return train_balanced_rf_binary()

st.title("MODELS AND RESULTS")

st.markdown("""
This section summarizes the main results obtained from two complementary modeling approaches:

1. **Multiclass classification** (Primary, Recidivist, Repeat offender) using Random Forest.
2. **Balanced binary classification** (Primary vs Recidivist) using Balanced Random Forest.
""")


# MULTICLASS MODEL BASE
res_base = get_base_results()
print(res_base)
with st.container(border= True):
    st.header("1. Base Model: Random Forest Multiclass")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{res_base['accuracy']:.2f}")
    c2.metric("Macro F1", f"{res_base['macro_f1']:.2f}")
    c3.metric("Recall (Recidivist)", f"{res_base['recall_by_class']['Reincidente']:.2f}")
    c4.metric("Recall (Repeat offender)", f"{res_base['recall_by_class']['Reiterante']:.2f}")
    st.caption(
    "Baseline: good overall accuracy, but low recall for minority classes (high false negatives)."
)
    st.markdown("""
    We start by training a **baseline Random Forest model** using intentionally conservative hyperparameters.  
    This model serves as a performance benchmark, allowing us to measure how much improvement we gain from hyperparameter tuning and model optimization.

    Baseline configuration:
    - 200 trees  
    - Unlimited depth  
    - Minimum 5 samples to split   

    The purpose of this baseline is not to maximize performance, but to establish a reliable reference point.""")

    st.subheader("Classification report")
    display_classification_report(res_base["y_test"], res_base["y_pred"], res_base["target_encoder"])
    st.caption(
        "The baseline model achieves satisfactory overall accuracy, but recall for minority classes "
        "(Recidivist, Repeat offender) is substantially lower."
    )

    st.subheader("Confusion matrix")
    tab_abs, tab_pct = st.tabs(["Counts", "Row-wise percentages"])
    cm = res_base["cm"]
    target_encoder = res_base["target_encoder"]

    class_names_en = [TARGET_LABELS_EN.get(c, c) for c in target_encoder.classes_]

    with tab_abs:
        fig_abs = plot_confusion_matrix(
            cm,
            class_names_en,
            cmap="Blues",
        )
        st.pyplot(fig_abs)

    with tab_pct:
        fig_pct = plot_confusion_matrix_normalized(
            cm,
            class_names_en,
            cmap="Blues",
        )
        st.pyplot(fig_pct)


    st.markdown("### Key observations from the baseline model")

    st.markdown("""
    - ✅ The model achieves **good performance** for the majority class (**Primary**), 
    with high precision and recall.  
    - ⚠️ **Recidivist** and **Repeat offender** classes show **low recall**, meaning many true cases 
    are misclassified as Primary.  
    - 🎯 This pattern reflects the **underlying class imbalance** in the dataset 
    (≈72% Primary, 19% Recidivist, 9% Repeat offender).  
    - 📌 From a policy and risk assessment perspective, **false negatives** in Recidivist/Repeat offender 
    are particularly critical.
    """)

    st.warning(
        "The baseline model tends to classify many Recidivists and Repeat offenders as Primary. "
        "This high rate of false negatives for minority classes motivates the need for rebalancing "
        "and hyperparameter optimization."
    )

    with st.expander("Read detailed baseline model interpretation"):
        st.markdown("""
    The baseline model achieved an overall accuracy of **75%**, which on the surface appears reasonable 
    for a multiclass classification problem of this complexity. However, a closer examination of the 
    class-specific metrics reveals a strong imbalance in performance between the majority class 
    (**Primary**) and the minority classes (**Recidivist** and **Repeat offender**).

    The underlying dataset is highly imbalanced (≈72% Primary, 19% Recidivist, 9% Repeat offender). 
    In models such as Random Forest, this imbalance tends to bias predictions toward the majority class, 
    since the model learns more from the dominant patterns and is exposed to fewer examples of 
    minority classes.

    For the **Primary** class, precision and recall are both high (precision ≈ 0.79, recall ≈ 0.94), 
    meaning the model correctly identifies most Primary cases and rarely misclassifies other individuals 
    as Primary.

    In contrast, the **Recidivist** and **Repeat offender** classes present much lower recall 
    (≈0.29 and ≈0.22, respectively). The model fails to detect a large proportion of true recidivists, 
    even though its precision is moderate (≈0.52–0.54) when it does predict these classes.

    This pattern indicates that the baseline model systematically misclassifies many Recidivists and 
    Repeat offenders as Primary offenders — that is, it produces a high number of false negatives for 
    the classes of greatest analytical interest.

    From a criminal justice or public policy perspective, false negatives are especially problematic, 
    as they correspond to individuals with prior offending trajectories being incorrectly categorized 
    as first-time offenders.

    Overall, the baseline model successfully captures broad population-level patterns, but its 
    sensitivity to minority classes remains insufficient, confirming the need for rebalancing strategies, 
    hyperparameter tuning, and potentially alternative model families.
        """)

    st.info("Next step: apply class balancing and regularization to improve recall on Recidivist/Repeat offender.")

st.markdown("---")

# MULTICLASS MODEL
res_multi = get_multiclase_results()
with st.container(border= True):
    st.header("2. Optimized Random Forest Multiclass Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",
              f"{res_multi['accuracy']:.2f}",
              delta=f"{res_multi['accuracy'] - res_base['accuracy']:+.2f}")
    c2.metric("Macro F1",
              f"{res_multi['macro_f1']:.2f}",
              delta=f"{res_multi['macro_f1'] - res_base['macro_f1']:+.2f}")
    c3.metric("Recall (Recidivist)",
              f"{res_multi['recall_by_class']['Reincidente']:.2f}",
              delta=f"{res_multi['recall_by_class']['Reincidente'] - res_base['recall_by_class']['Reincidente']:+.2f}")
    c4.metric("Recall (Repeat offender)",
              f"{res_multi['recall_by_class']['Reiterante']:.2f}",
              delta=f"{res_multi['recall_by_class']['Reiterante'] - res_base['recall_by_class']['Reiterante']:+.2f}")
    st.markdown("""
    Hyperparameter analysis

    A sensitivity analysis was conducted to examine the effect of key Random Forest hyperparameters (n_estimators, min_samples_split, min_samples_leaf) on AUC–ROC performance.

    The analysis revealed diminishing returns beyond 300 trees, and optimal regularization around min_samples_split = 10 and min_samples_leaf = 3. These values were therefore selected to balance predictive performance and model complexity.

    Final model evaluation was performed on a held-out test set.

    We also added class_weight="balanced" to compensate for class imbalance and improve the model’s ability to detect Recidivist and Repeat offender cases """)

    display_classification_report(res_multi["y_test"], res_multi["y_pred"], res_multi["target_encoder"])

    st.subheader("Confusion matrix")

    cm = res_multi["cm"]
    target_encoder = res_multi["target_encoder"]

    tab_abs, tab_pct = st.tabs(["Counts", "Row-wise percentages"])

    class_names_en = [
        TARGET_LABELS_EN.get(c, c)
        for c in target_encoder.classes_
    ]

    with tab_abs:
        fig_abs = plot_confusion_matrix(
            cm,
            class_names_en,
            cmap="Blues",
        )
        st.pyplot(fig_abs)

    with tab_pct:
        fig_pct = plot_confusion_matrix_normalized(
            cm,
            class_names_en,
            cmap="Blues",
        )
        st.pyplot(fig_pct)

    st.markdown("### Key observations from the optimized model")

    st.markdown("""
    - 🔄 **Overall accuracy** decreases slightly (**0.75 → 0.71**), as a consequence of explicitly 
    prioritizing minority-class performance.  
    - 📈 **Recall for Repeat offender** improves strongly (**0.22 → 0.57**), and **Recidivist** recall 
    also increases (**0.29 → 0.47**). The model now correctly identifies many more true recidivists.  
    - ⚠️ **Precision for minority classes** remains moderate (≈0.37–0.45), indicating that some false 
    positives persist.  
    - ⚖️ The model therefore trades a small amount of global accuracy for a more **balanced and fair** 
    treatment of all classes.
    """)

    st.success(
        "From a policy and risk-assessment perspective, improving the detection of Recidivists and "
        "Repeat offenders (reducing false negatives) is more valuable than maximizing overall accuracy "
        "driven by the majority class. The optimized model is therefore better aligned with the goals "
        "of recidivism prediction."
    )

    with st.expander("Read detailed optimized model interpretation"):
        st.markdown("""
    The optimized model shows a slight decrease in overall accuracy (from **0.75 → 0.71**).  
    However, this reduction is expected and acceptable, as the model has been explicitly tuned to 
    improve performance on minority classes—**Recidivist** and **Repeat offender**—rather than 
    maximizing global accuracy.

    The improvements in sensitivity are considerable:

    - **Repeat offender:** recall increases from **0.22 → 0.57**  
    - **Recidivist:** recall increases from **0.29 → 0.47**

    This means the optimized model is now able to identify many more true cases of recidivism, 
    reducing the number of false negatives that were prevalent in the baseline model.

    Precision values for minority classes remain in the **0.37–0.45** range. This indicates that, 
    although the model is better at detecting recidivists, it still produces some false positives.

    Overall, the optimized Random Forest exhibits a more balanced behavior: it improves recall for 
    minority classes, reduces false negatives, and accepts a small loss in overall accuracy in 
    exchange for greater fairness and substantive relevance.

    From a correctional system and public policy perspective, this is a desirable trade-off: accurately 
    identifying individuals with prior offending trajectories is more valuable than slightly improving 
    global accuracy driven mostly by the majority class (Primary offenders).
        """)

st.markdown("---")

# BINARY BALANCED MODEL
res_bin = get_binario_results()

with st.container(border=True):
    st.header("3. Binary Balanced Random Forest")

    st.markdown("### Why a binary formulation?")
    st.caption(
        "We group **Recidivist** and **Repeat offender** into a single class (**Recidivism**) because both represent "
        "a repeated offending trajectory. The distinction between them is mainly legal/quantitative, while the "
        "operational question is often: **will this person reoffend (yes/no)?**"
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balanced Acc", f"{res_bin['balanced_accuracy']:.2f}")
    c2.metric("Recall (Recidivist)", f"{res_bin['recall_pos']:.2f}")
    c3.metric("Precision (Recidivist)", f"{res_bin['precision_pos']:.2f}")
    c4.metric("F1 (Recidivist)", f"{res_bin['f1_pos']:.2f}")

    st.caption(
        "Default predictions use the standard threshold of **0.50** on P(Recidivist). "
        "The slider below shows how changing this threshold changes the trade-off."
    )

    # Defaul 0.5 threshold results
    st.subheader("Classification report (default threshold = 0.50)")
    display_classification_report(res_bin["y_test"], res_bin["y_pred"], res_bin["target_encoder"])

    st.subheader("Confusion matrix (default threshold = 0.50)")
    tab_abs, tab_pct = st.tabs(["Counts", "Row-wise percentages"])

    cm = res_bin["cm"]
    class_names = res_bin["target_encoder"].classes_

    with tab_abs:
        fig_abs = plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            cmap="Greens",
            figsize=(3.2, 2.6),
        )
        st.pyplot(fig_abs)

    with tab_pct:
        fig_pct = plot_confusion_matrix_normalized(
            cm=cm,
            class_names=class_names,
            cmap="Greens",
            figsize=(3.2, 2.6),
        )
        st.pyplot(fig_pct)

    st.markdown("""
- Overall accuracy is slightly lower than the baseline, but performance is **more balanced** across classes.  
- The model achieves **higher recall for recidivism**, meaning fewer false negatives among recidivists.  
- This comes with a moderate increase in false positives, which can be acceptable in operational settings where 
  missing true recidivists is considered more costly than flagging some primaries.
""")

    st.markdown("---")

    # ✅ Threshold tuning (separado y claro)
    st.subheader("Threshold tuning (interactive)")

    st.info(
        "Lower threshold → higher recall (detect more recidivists) but more false positives. "
        "Higher threshold → higher precision but more false negatives."
    )

    thr = st.slider(
        "Decision threshold for predicting **Recidivist**",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    y_true = res_bin["y_test"].to_numpy() if hasattr(res_bin["y_test"], "to_numpy") else np.asarray(res_bin["y_test"])
    proba_pos = res_bin["y_proba"][:, res_bin["pos_label"]]

    y_pred_thr = (proba_pos >= thr).astype(int)
    pos_label = res_bin["pos_label"]
    y_true_pos = (y_true == pos_label).astype(int)

    precision_t = precision_score(y_true_pos, y_pred_thr, zero_division=0)
    recall_t = recall_score(y_true_pos, y_pred_thr, zero_division=0)
    f1_t = f1_score(y_true_pos, y_pred_thr, zero_division=0)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Precision @ threshold", f"{precision_t:.2f}")
    d2.metric("Recall @ threshold", f"{recall_t:.2f}")
    d3.metric("F1 @ threshold", f"{f1_t:.2f}")
    d4.metric("% predicted Recidivist", f"{y_pred_thr.mean():.1%}")

    # Confusion matrix @ threshold
    cm_thr = confusion_matrix(y_true_pos, y_pred_thr, labels=[0, 1])
    st.caption("Confusion matrix @ threshold (rows: true, cols: predicted)")
    st.pyplot(
        plot_confusion_matrix(
            cm=cm_thr,
            class_names=["Primary", "Recidivist"],
            cmap="Greens",
            figsize=(3.2, 2.6),
        )
    )
    st.markdown("""
### Final interpretation

The binary balanced model provides a **complementary and operationally oriented view** of recidivism risk.
By collapsing *Repeat offender* into the *Recidivist* category, the task shifts from fine-grained classification
to the detection of **any prior offending trajectory**, which is often the primary concern in applied settings.

The use of **Balanced Random Forest** substantially improves sensitivity to the positive class compared to
the multiclass baseline, reducing the number of false negatives among recidivists.

Crucially, the interactive threshold analysis shows that model performance is **not fixed**: the decision
threshold allows practitioners to explicitly control the trade-off between **recall** and **precision**.
Lower thresholds favor the detection of recidivism at the cost of more false positives, while higher thresholds
prioritize conservative classifications.

Overall, this formulation illustrates that effective recidivism modeling depends not only on the algorithm,
but also on **how predictions are operationalized**, making threshold selection a policy-relevant decision
rather than a purely technical one.
""")

st.markdown("---")

st.markdown("## Overall conclusion")
st.markdown("""
This study developed and compared multiple Random Forest models to predict recidivism using prison administrative data.

- The **multiclass model** (Primary / Recidivist / Repeat offender) provides a **more detailed** view of the phenomenon, 
  which is useful for descriptive analysis and differentiated policy design.  
- The **binary model** (Recidivism vs No recidivism) improves **detection capacity** and offers a clearer operational tool 
  for prevention-oriented decisions.

In summary, both approaches are **complementary**: multiclass supports strategic understanding, while binary supports 
practical risk detection and resource allocation.
""")

   