import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from functions.functions import calculate_asociations, translate_df_for_display, var_label
from functions.translations import VARIABLE_LABELS
from functions.utils import truncar_labels_ejes
from functions.chart_utils import pie_chart, barh_count, bar_count, hist_chart, plot_bivariate_categorical
from data.data_loader import load_data

sns.set_theme(style="whitegrid")
st.title("📊 Data exploration")

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_translated_df(df):
    return translate_df_for_display(df)

df = get_data()
df_en = get_translated_df(df)  

# Sections and tabs for the EDA
tab_overview, tab_target, tab_univ, tab_biv, tab_assoc, tab_num, tab_summary = st.tabs([
    "Overview",
    "Target & imbalance",
    "Univariate",
    "Bivariate",
    "Associations",
    "Numeric by target",
    "Summary",
])

with tab_overview:
    st.subheader("General description of the dataset")

    st.write(f"Number of rows: **{df.shape[0]}**")
    st.write(f"Number of columns: **{df.shape[1]}**")

    st.markdown("### View of the first rows (Spanish)")
    st.dataframe(df.head())

    with st.expander("Statistical description"):
        st.dataframe(df.describe())

with tab_target:
    st.subheader("The target variable")
    st.markdown("""
    **In our dataset from Argentina’s SNEEP (National System of Statistics on Sentence Execution), the target variable `es_reincidente_descripcion` classifies incarcerated individuals into three categories based on their prior criminal history and recidivism status:**

    - **Primario/a (First-time offender):** Individuals with **no prior convictions or imprisonment**, incarcerated for the first time.

    - **Reincidente (Recidivist, Art. 50 of the Argentine Criminal Code):** Individuals who **have been previously convicted** and are now serving a new sentence following another offense.

    - **Reiterante (Repeat offender):** Individuals with **multiple prior convictions**, reflecting a pattern of repeated offending over time.
    """)

    st.markdown("---")
    st.subheader("Imbalance of the target variable")

    if "es_reincidente_descripcion" in df.columns:

        labels_map = {
            "Primario/a": "Primary",
            "Reincidente (art. 50 CP)": "Recidivist",
            "Reiterante": "Repeat offender",
        }

        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("""
    The target variable `es_reincidente_descripcion` is highly imbalanced.

    The imbalance ratio (majority/minority class) is **8.37×**, meaning there are more than eight first-time offenders for every repeat offender.

    **Class distribution:**

    - **72.15%** — First-time offender (Primary)  
    - **19.23%** — Recidivist (Art. 50 Criminal Code)  
    - **8.62%** — Repeat offender  

    This reflects that, in the Argentine prison system, most incarcerated individuals do not have prior recidivism records.
            """)

        with col_right:
            fig = pie_chart(
                df["es_reincidente_descripcion"],
                labels_map=labels_map,
                figsize=(4, 4),
                dpi=220,
                label_fontsize=12,
                pct_fontsize=10,
            )
            st.pyplot(fig)

        st.markdown("""
    ### Implications for Modeling

    - The target variable is **highly imbalanced**.  
    - A model may simply predict the majority class (“Primary”) and still achieve high accuracy.  
    - It is important to use **balancing techniques** and **alternative evaluation metrics** (F1-score, recall per class) to correctly detect minority classes.
        """)

    else:
        st.warning("No se encontró la columna 'es_reincidente_descripcion' en el dataset.")


with tab_univ:
    st.subheader("Univariate descriptive analysis")
    st.markdown("## 📊 Univariate analysis of key variables")

    FIG_SIZE = (5, 4)
    st.markdown("### 1. Demographic characteristics")

    row1_col1, row1_col2 = st.columns(2)

    # Gender
    with row1_col1:
        if "genero_descripcion" in df_en.columns:
            st.markdown("#### Gender")

            fig = pie_chart(
                df_en["genero_descripcion"],
                title="Gender distribution",
                figsize=FIG_SIZE,
                labels_map=var_label("genero_descripcion")
            )
            st.pyplot(fig)

    # Age
    with row1_col2:
        if "edad" in df.columns:
            st.markdown("#### Age")

            fig = hist_chart(
                df["edad"],
                bins=25,
                title="Age distribution",
                xlabel="Age",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)
    st.markdown(""" 
    The age distribution is right-skewed, concentrated approximately between **25 and 40 years old**, with a median close to **34 years** and few extreme cases above 70. This suggests a predominantly **young adult** prison population.
    
    Around **96%** of the individuals are **male**, reflecting the well-known overrepresentation of men in the prison system.
    The majority are **single** (around **81%**), followed by **cohabiting** and **married** individuals, each representing roughly **7%** of the sample.
    """)

    row2_col1, row2_col2 = st.columns(2)

    st.markdown("### 2. Education and previous work situation")

    row2_col1, row2_col2 = st.columns(2)

    # Education level
    with row2_col1:
        if "nivel_instruccion_descripcion" in df_en.columns:
            st.markdown("#### Education level")

            fig = barh_count(
                df_en["nivel_instruccion_descripcion"],
                labels_map=var_label("nivel_instruccion_descripcion"),
                title="Education level",
                xlabel="Frequency",
                ylabel="Education level",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)

    # Previous work situation
    with row2_col2:
        if "ultima_situacion_laboral_descripcion" in df_en.columns:
            st.markdown("#### Previous work situation")

            fig = barh_count(
                df_en["ultima_situacion_laboral_descripcion"],
                labels_map=var_label("ultima_situacion_laboral_descripcion"),
                title="Previous work situation",
                xlabel="Frequency",
                ylabel="Work situation",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)
    st.markdown("""
    **Education level**  
    - Approximately **98%** do **not** have tertiary or university education (either completed or incomplete).  
    - Around **10%** report **no formal education**.  
    - About **32%** have completed only **primary education**, and roughly **25%** have **incomplete secondary education**.  

    Overall, this points to a **low educational profile**, consistent with criminological literature on educational vulnerability in incarcerated populations.

    **Employment status**
    - Approximately 40% of individuals reported part-time employment prior to incarceration, while nearly 30% had no formal employment.
    Only 26% reported full-time employment, indicating limited and unstable labor market attachment.

    **Vocational and professional skills:**  
    - Around **50%** reported possessing a **manual trade**, whereas **14%** indicated having a 
    **professional qualification**. This suggests a predominantly low- or semi-skilled labor profile.""")
    st.markdown("### 3. Main offense and disciplinary behavior")

    row3_col1, row3_col2 = st.columns(2)

    # Main offense (Top 10)
    with row3_col1:
        if "delito1_descripcion" in df_en.columns:
            st.markdown("#### Main offense (Top 10)")

            fig = barh_count(
                df_en["delito1_descripcion"],
                labels_map=var_label("delito1_descripcion"),
                title="Main offense (Top 10)",
                xlabel="Frequency",
                ylabel="Offense",
                top=10,
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)

    # Disciplinary infractions
    with row3_col2:
        if "tipo_infraccion_disciplinaria_descripcion" in df_en.columns:
            st.markdown("#### Disciplinary infractions")

            fig = barh_count(
                df_en["tipo_infraccion_disciplinaria_descripcion"],
                labels_map=var_label("tipo_infraccion_disciplinaria_descripcion"),
                title="Disciplinary infractions",
                xlabel="Frequency",
                ylabel="Infraction type",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)
    st.markdown("""- **Primary offense:**  
    - **30%** correspond to **robbery or attempted robbery**,  
    - **20%** to **sexual crimes**,  
    - **12%** to **intentional homicide**.  
    Additionally, **90%** have only **one offense** recorded, indicating low prevalence of multi-offense histories.

    - **Disciplinary infractions:**
    **85%** of individuals have **no recorded disciplinary infractions**.""")

    st.markdown("### 4. Institutional conduct and sentence length")

    row4_col1, row4_col2 = st.columns(2)

    # Institutional conduct
    with row4_col1:
        if "calificacion_conducta_descripcion" in df_en.columns:
            st.markdown("#### Institutional conduct")

            fig = barh_count(
                df_en["calificacion_conducta_descripcion"],
                labels_map=var_label("calificacion_conducta_descripcion"),
                title="Institutional conduct",
                xlabel="Frequency",
                ylabel="Conduct rating",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)

    # Sentence length (years)
    with row4_col2:
        if "duracion_condena_anios" in df.columns:
            st.markdown("#### Sentence length (years)")

            fig = hist_chart(
                df["duracion_condena_anios"],
                bins=30,
                title="Sentence length (years)",
                xlabel="Years",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)
    st.markdown("""
    - **Behavioral classification:**  
    Most individuals are categorized as having **“Exemplary”** institutional conduct.
    - **Sentence duration:**  
    The distribution is **right-skewed**, with a median near **7 years**. Outlier values correspond to **life sentences**.
    - **Type of sentence:**  
    Approximately **4.6%** are serving **life imprisonment**.""")
    st.markdown("### 5. Participation in programs")

    prog_col1, prog_col2 = st.columns(2)

    # Pre-release program
    with prog_col1:
        if "participa_programa_pre_libertad" in df.columns:
            st.markdown("#### Pre-release program")

            pre_labels = {0: "No", 1: "Yes"}

            fig = bar_count(
                df["participa_programa_pre_libertad"],
                labels_map=pre_labels,
                title="Participation in pre-release program",
                xlabel="Pre-release",
                ylabel="Frequency",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)

    # Sports activities
    with prog_col2:
        if "participacion_actividades_deportivas" in df.columns:
            st.markdown("#### Sports activities")

            sports_labels = {0: "No", 1: "Yes"}

            fig = bar_count(
                df["participacion_actividades_deportivas"],
                labels_map=sports_labels,
                title="Participation in sports activities",
                xlabel="Sports activities",
                ylabel="Frequency",
                figsize=FIG_SIZE,
            )
            st.pyplot(fig)
    st.markdown("""
    - **Educational programs:**  
    - **50%** do **not** participate in any educational activity.  
    - **32%** attend **primary or secondary-level educational programs**.

    - **Labor programs inside the institution:**  
    **27%** participate in **in-prison labor programs**, indicating moderate engagement in work activities.

    - **Pre-release programs:**  
    Only **12%** are enrolled in **pre-release programs**, reflecting limited access to reintegration pathways.

    - **Sports and recreational activities:**  
    Participation is notably high (**71.6%**), consistent with institutional policies promoting physical activity
    and structured recreation.""")

    if "ultima_provincia_residencia_descripcion" in df.columns:
        st.markdown("### 6. Last province of residence (Top 10)")

        fig = barh_count(
            df["ultima_provincia_residencia_descripcion"],
            title="Last province of residence (Top 10)",
            xlabel="Frequency",
            ylabel="Province",
            top=10,
            figsize=(8, 4),
        )
        st.pyplot(fig)

    st.info("""
    For clarity and space reasons, not all variables and plots are displayed in this dashboard.
    The complete exploratory data analysis, including additional variables and full code,
    is available in the project repository:
    [View full EDA code](https://tu-link-aqui.com)
    """)
    st.markdown("---")

with tab_biv:
    st.subheader("Bivariate descriptive analysis")

    st.markdown("""
    This section explores how key categorical variables relate to the target variable 
    `es_reincidente_descripcion` (recidivism status).  
    The objective is to identify patterns that may suggest differential profiles for primary offenders, 
    recidivists, and repeat offenders, and to guide feature selection for the predictive models.
    """)

    st.info("""
**Summary:**  
The bivariate analysis shows that recidivism is strongly associated with structural disadvantage,
criminal history accumulation, and institutional trajectories.  
While several variables display high predictive potential, many reflect prior institutional decisions,
requiring careful selection and ethical consideration before inclusion in predictive models.
""")

    vars_categoricas_importantes = [
        "delito1_descripcion",
        "nivel_instruccion_descripcion",
        "ultima_situacion_laboral_descripcion",
        "calificacion_conducta_descripcion",
        "ultima_provincia_residencia_descripcion",
        "tiene_periodo_progresividad_descripcion",
        "tipo_infraccion_disciplinaria_descripcion",
        "participa_programa_pre_libertad",
        "participacion_actividades_deportivas",
    ]

    st.markdown("### 📊 Relationship between categorical variables and recidivism")

    if vars_categoricas_importantes:

        # Usamos las etiquetas en inglés para mostrar en el select
        var_sel = st.selectbox(
            "Select a categorical variable to visualize its relationship with recidivism:",
            vars_categoricas_importantes,
            format_func=lambda x: VARIABLE_LABELS.get(x, x)
        )

        # df_disp: dataframe ya traducido para visualización
        fig_hm, fig_stack = plot_bivariate_categorical(
            df_en,
            var_sel,
            "es_reincidente_descripcion",
            VARIABLE_LABELS
        )

        tab_hm, tab_stack = st.tabs([
            "Heatmap (row-wise percentages)",
            "Stacked bar chart (row-wise percentages)"
        ])

        var_label_en = VARIABLE_LABELS.get(var_sel, var_sel)

        with tab_hm:
            st.markdown(
                f"#### Heatmap: distribution of recidivism across categories of **{var_label_en}**"
            )
            ax_hm = fig_hm.axes[0]
            truncar_labels_ejes(ax_hm, max_chars_x=10, max_chars_y=30)
            st.pyplot(fig_hm)

            st.caption("""
            The heatmap shows, for each category of the selected variable, the percentage distribution 
            across the three recidivism classes (Primary, Recidivist, Repeat offender).  
            Darker cells indicate higher proportions and may reveal categories with a stronger association 
            with recidivism.
            """)

        with tab_stack:
            st.markdown(
                f"#### Stacked bar chart: composition of recidivism within **{var_label_en}**"
            )
            ax_stack = fig_stack.axes[0]
            truncar_labels_ejes(ax_stack, max_chars_x=12, max_chars_y=30, rot_x=25)
            st.pyplot(fig_stack)

            st.caption("""
            The stacked bar chart displays, for each category, the proportion of cases belonging to each 
            recidivism class. This representation is useful to compare how the relative weight of recidivism 
            changes across categories, even when the absolute number of observations differs.
            """)

    else:
        st.warning("No categorical variables are available for this analysis.")
    st.subheader("Key findings from the bivariate analysis")

    st.markdown("""
    The bivariate analysis reveals that recidivism is **not randomly distributed** across the prison population.
    Instead, it shows systematic associations with sociodemographic characteristics, criminal history, and
    institutional trajectories.

    **Territory and institutional context**
    - The distribution of recidivism varies across provinces and detention facilities.
    - Some establishments concentrate higher proportions of repeat offenders, likely reflecting
    institutional classification practices rather than causal effects.

    **Education, employment, and social vulnerability**
    - Lower educational attainment is consistently associated with higher levels of recidivism.
    - Individuals with incomplete schooling or no formal education exhibit higher proportions of
    recidivist and repeat-offender trajectories.
    - Previous employment status shows weaker but still relevant differences, suggesting unstable labor
    attachment among recidivists.

    **Criminal profile**
    - Property crimes (robbery and theft) display higher recidivism rates than violent crimes against life.
    - The presence of multiple offenses (secondary crimes) is strongly associated with recidivism,
    reflecting the cumulative nature of criminal trajectories.

    **Institutional behavior and programs**
    - Poor institutional conduct and the presence of disciplinary infractions are associated with
    higher recidivism rates.
    - Participation in educational, labor, and pre-release programs is associated with lower proportions
    of recidivism, suggesting a potential protective role.
    - However, participation may be partially driven by prior good behavior, introducing possible
    selection bias.

    **Methodological caution**
    - Several highly predictive variables capture institutional decisions or post-classification behavior.
    - These variables are informative for descriptive analysis but may introduce data leakage if used
    uncritically in predictive models.
    """)


    

with tab_assoc:

    @st.cache_data
    def get_asociations(df, vars_cat, target):
        return calculate_asociations(df, vars_cat, target)

    asociations = get_asociations(
        df,
        vars_categoricas_importantes,
        "es_reincidente_descripcion"
    )
    asociations_display = asociations.copy()

    asociations_display["Variable"] = asociations_display["Variable"].map(
        lambda x: VARIABLE_LABELS.get(x, x)
    )

    st.subheader("Association with the target (Chi-square and Cramér’s V)")
    st.markdown("""
The table above summarizes the strength of association between selected categorical variables
and the target variable (*recidivism status*), using the Chi-square test and Cramér’s V.

All variables show **statistically significant associations** (p < 0.001), which is expected given
the large sample size. Therefore, interpretation focuses primarily on **effect size** (Cramér’s V),
rather than statistical significance alone.
""")


    if asociations.empty:
        st.warning("No associations could be computed.")
    else:
        st.dataframe(asociations_display)

with tab_num:
    st.subheader("Numeric variables by recidivism status")
    num_cols = ["edad", "duracion_condena_anios"]
    num_cols = [c for c in num_cols if c in df.columns]

    if num_cols:
        n = len(num_cols)
        rows = math.ceil(n / 2)

        fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 4))
        axes = np.array(axes).reshape(-1)  
        colores = ['#1f77b4', '#ffcc00', '#d62728'] 

        for ax, col in zip(axes, num_cols):
            sns.boxplot(
                x="es_reincidente_descripcion",
                y=col,
                data=df,
                ax=ax,
                palette=colores
            )
            ax.set_title(f"{col.replace('_', ' ').capitalize()} según reincidencia", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel(col.replace('_', ' ').capitalize())
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=10)

        for i in range(n, len(axes)):
            axes[i].axis("off")

        st.pyplot(fig)
        st.markdown("""
        **Age**

        - The age distributions across the three groups are largely overlapping.
        - This suggests that age alone is **not a strong discriminator** of recidivism status,
        although it may still contribute marginal predictive value when combined with other variables.
        """)
    st.markdown("""
**Sentence length (years)**

- Sentence duration shows a **right-skewed distribution** in all groups, with a small number
  of very long sentences (including life sentences).
- The substantial overlap between distributions indicates that sentence length
  is more reflective of **judicial outcomes** than of individual recidivism risk per se.
""")

        
with tab_summary:
    st.subheader("EDA summary")
    st.markdown("""
Key takeaways from exploration:

- The target is **strongly imbalanced**, so accuracy alone can be misleading.
- Several categorical variables show meaningful association with recidivism (Chi² / Cramér’s V).
- Numeric variables (age, sentence length) show different distributions across groups.

These insights motivate the modeling strategy used in the next section (class balancing, macro metrics, and complementary binary modeling).
""")
