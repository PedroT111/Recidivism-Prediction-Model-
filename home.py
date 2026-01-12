import streamlit as st

st.set_page_config(
    page_title="Reincidencia penitenciaria - SNEEP",
    layout="wide",
)

st.title("Prediction of Prison Recidivism in Argentina")
st.markdown("""

Academic project based on data from the National Statistics System on Sentence Execution (SNEEP).

The goal is to explore the extent to which certain sociodemographic, criminal, and institutional-trajectory variables are related to recidivism among incarcerated individuals, and to train classification models that can approximate that probability.

---
""")
st.subheader("Application Content")

st.markdown("""
- 📊 **Data exploration:** description of the dataset, distributions, and initial relationships.
- 🧠 **Modeling and results:** Multiclass Random Forest and Binary Balanced Random Forest.
- ⚖️ **Ethics and conclusions:** discussion on biases, limitations, and future lines of work.
""")

st.subheader("Repository and notebook")

st.markdown("""
- 💻 Full code and notebook on Google Colab: (https://colab.research.google.com/drive/1RyjXitjkVi-sdDk3ASsi129QJMKgEMel#scrollTo=EOrL2Ksdiy6o)
---
""")


st.markdown(""" 
# Project Overview

### **1. Problem Context**

Recidivism prediction is a critical challenge in the Argentine criminal justice and correctional system. Understanding the factors associated with repeated offending supports evidence-based decision making, more equitable interventions, and improved reintegration policies for individuals deprived of liberty.

---

### **2. Dataset Description (SNEEP – Argentina)**

The dataset used in this project comes from **SNEEP** (National System of Statistics on Sentence Execution).  
It contains demographic, legal, and institutional information about individuals incarcerated in the national prison system.

- The data represent a specific reporting year.  
- Variables may include demographic characteristics, criminal history, sentence information, and institutional records.  
- Like all administrative datasets, it reflects the limitations and structure of the system that produces it.

---

### **3. Target Variable: Offender Status**

The target variable categorizes individuals according to their prior criminal history, following the definitions used in SNEEP:

- **Primary (Primario/a):** No previous convictions or imprisonment.  
- **Repeat (Reiterante):** Multiple prior contacts with the justice or penal system.  
- **Reoffender (Reincidente):** A new offense committed after a previous conviction.

This is a **multi-class classification problem**, where the model must assign individuals to one of these three categories.

---

### **4. Project Objective**

The main objective of this work is to develop a classification model capable of predicting an individual’s offender status based on their available features.  
The exploratory data analysis (EDA) that follows helps identify patterns, potential risk factors, data quality issues, and model-relevant relationships.



""")
