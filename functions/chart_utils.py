import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def setup_matplotlib():
    mpl.rcParams.update({
        # Nitidez
        "figure.dpi": 200,
        "savefig.dpi": 200,

        # Tipos de letra
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,

        # Bordes y grid suaves
        "axes.edgecolor": "#999999",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,

        # Leyendas
        "legend.fontsize": 10,
    })


def pie_chart(
    series: pd.Series,
    labels_map: dict | None = None,
    title: str | None = None,
    figsize=(4, 4),
    dpi=200,
    label_fontsize=12,
    pct_fontsize=10,
    min_pct_label=3
):

    dist_abs = series.value_counts(dropna=False)
    total = dist_abs.sum()

    if labels_map is not None:
        labels = dist_abs.rename(index=labels_map).index
    else:
        labels = dist_abs.index

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    wedges, texts, autotexts = ax.pie(
        dist_abs,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= min_pct_label else "",
        startangle=90,

    )

    
    for t, val in zip(texts, dist_abs):
        pct = val / total * 100
        if pct < min_pct_label:
            t.set_text("")  
        else:
            t.set_fontsize(label_fontsize)

    for a in autotexts:
        a.set_fontsize(pct_fontsize)

    ax.axis("equal")

    if title:
        ax.set_title(title)

    return fig

def barh_count(
    series: pd.Series,
    labels_map: dict | None = None,
    title: str | None = None,
    top: int | None = None,
    xlabel: str = "Frequency",
    ylabel: str | None = None,
    figsize=(5, 4),
    dpi=200,
):
    counts = series.value_counts(dropna=False)

    if top is not None:
        counts = counts.head(top)

    if labels_map is not None:
        counts = counts.rename(index=labels_map)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    counts.plot(kind="barh", ax=ax)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def bar_count(
    series: pd.Series,
    labels_map: dict | None = None,
    title: str | None = None,
    xlabel: str = "Category",
    ylabel: str = "Frequency",
    rotation: int = 0,
    figsize=(4, 4),
    dpi=200,
):
    counts = series.value_counts(dropna=False)

    if labels_map is not None:
        counts = counts.rename(index=labels_map)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    counts.plot(kind="bar", ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def hist_chart(
    series: pd.Series,
    bins: int = 30,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Frequency",
    figsize=(5, 4),
    dpi=200,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(series.dropna(), bins=bins)

    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def limit_categories(df, var, max_cats=8):
    counts = df[var].value_counts()

    if len(counts) <= max_cats:
        return df[var]  # no hace falta limitar

    top_cats = counts.index[:max_cats-1]  
    nueva_columna = df[var].apply(lambda x: x if x in top_cats else "Otros")

    return nueva_columna

def plot_bivariate_categorical(df, var, target):
    serie_limitada = limit_categories(df, var, max_cats=7)

    tab = pd.crosstab(serie_limitada, df[target], normalize="index") * 100
    # Heatmap
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.heatmap(tab, annot=True, fmt=".1f", cmap="Greens", ax=ax1)
    ax1.set_title(f"Distribución porcentual de {target} dentro de {var}")
    ax1.set_xlabel(target)
    ax1.set_ylabel(var)
    
    # Barras apiladas
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    tab.plot(kind="bar", stacked=True, ax=ax2, colormap="Greens")
    ax2.set_ylabel("% dentro de cada categoría")
    ax2.set_title(f"Reincidencia (%) según {var}")
    ax2.legend(title=target, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.tick_params(axis="x", rotation=30)



    return fig1, fig2

