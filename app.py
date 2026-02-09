# app.py
import streamlit as st
from functions.chart_utils import setup_matplotlib

setup_matplotlib()

st.set_page_config(
    page_title="Reincidencia penitenciaria - SNEEP",
    layout="wide",
)

st.title("Prediction of Prison Recidivism in Argentina")
st.markdown("""
Academic project based on data from the **National System of Statistics on Sentence Execution (SNEEP)**.

The goal is to explore how sociodemographic, criminal, and institutional trajectory variables
relate to recidivism among incarcerated individuals, and to evaluate classification models
that can approximate this risk in an exploratory setting.

> This project is **academic and non-operational**. Models are intended for analysis and discussion,
> not for automated decision-making.
---
""")

st.subheader("Application Content")

st.markdown("""
- 📊 **Data exploration:** dataset description, distributions, and initial relationships.  
- 🧠 **Modeling and results:** Multiclass Random Forest and Balanced Binary Random Forest.  
- ⚖️ **Ethics and conclusions:** discussion of biases, limitations, and future research directions.
""")

st.subheader("Code and reproducibility")

st.markdown("""
- 💻 Full code and exploratory notebook (Google Colab):  
  https://colab.research.google.com/drive/1RyjXitjkVi-sdDk3ASsi129QJMKgEMel
---
""")


st.markdown(""" 
# Project Overview

### **1. Problem context**

Recidivism prediction is a critical challenge in the Argentine criminal justice and correctional system.
Understanding the factors associated with repeated offending can support evidence-based policy discussions,
more equitable interventions, and improved reintegration strategies for individuals deprived of liberty.

---

### **2. Dataset Description (SNEEP – Argentina)**

The dataset used in this project comes from **SNEEP** (National System of Statistics on Sentence Execution).  
It contains demographic, legal, and institutional information about individuals incarcerated in the national prison system.

- The data represent a specific reporting year.  
- Variables may include demographic characteristics, criminal history, sentence information, and institutional records.  
- Like all administrative datasets, it reflects the limitations and structure of the system that produces it.

---

### **3. Target variable: offender status**

The target variable categorizes individuals according to their prior criminal history, following SNEEP definitions:

- **Primary:** no previous convictions or imprisonment.  
- **Recidivist:** a new offense committed after a previous conviction.  
- **Repeat offender:** multiple prior contacts with the criminal justice system.        

    This formulation supports a **multiclass classification** approach.  
    Additionally, a **binary formulation (recidivism vs. no recidivism)** is explored to support operationally oriented analysis.
---

### **4. Project objective**

The main objective of this work is to develop classification models capable of approximating
an individual’s offender status based on available features.

The exploratory data analysis (EDA) that follows aims to identify patterns, potential risk factors,
data quality issues, and relationships relevant for modeling, while maintaining a critical and ethical perspective.
""")
