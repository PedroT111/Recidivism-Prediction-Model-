# pages/4_Etica_y_conclusiones.py
import streamlit as st

st.title("Ethics, limitations and conclusions")

st.markdown("""
The use of machine learning models in criminal justice and prison systems raises
significant ethical, legal and political challenges.  
This project approaches **recidivism prediction** from an **exploratory and academic perspective**,
which requires several important considerations.
""")
st.markdown("### 1. Data bias and representativeness")
st.markdown("""
- The dataset reflects **previous institutional decisions** (sentencing, detention,
  disciplinary classifications), which may embed **structural inequalities**
  related to territory, social class, gender or other factors.
- Models trained on such data inevitably **inherit these biases** and may even
  amplify them if used uncritically.
""")
st.markdown("### 2. Risks of operational use")
st.markdown("""
- This model was developed **exclusively for academic and research purposes**.
- It has **not** been validated for real-world use in:
  - sentence progression decisions,
  - parole or temporary release,
  - individual risk assessment.
- Using predictive models as automated decision tools may lead to
  **algorithmic injustice** if applied without safeguards.
""")
st.markdown("### 3. Interpretability and transparency")
st.markdown("""
- Random Forest models offer reasonable performance and feature importance estimates,
  but they are **not fully transparent** to all stakeholders
  (judges, prison staff, incarcerated individuals, legal defense).
- Any practical use would require:
  - clear and accessible explanations,
  - detailed documentation,
  - and the possibility of **external auditing**.
""")
st.markdown("### 4. Future work")
st.markdown("""
Possible extensions of this work include:

- Evaluating **algorithmic fairness**, for example by analyzing performance across
  provinces, gender, or other sensitive attributes.
- Exploring more **interpretable models** (simple decision trees, penalized linear models)
  as complements to higher-performing approaches.
- Working in **interdisciplinary teams** involving criminologists, prison professionals,
  human rights organizations, and people with lived experience of the penal system.
""")
st.markdown("### 5. Final remarks")
st.markdown("""
This project shows that prison administrative data can be used to approximate
patterns of recidivism using machine learning techniques. However:

> The main conclusion is not that *“the model performs well”*, but that **any attempt to model penal phenomena must be accompanied by strong ethical, legal and political reflection**.

The Streamlit application is designed to make visible both the **quantitative results**
and the **critical questions** that inevitably arise when working in this domain.
""")
