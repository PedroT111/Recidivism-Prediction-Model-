def truncar_label(texto, max_chars=20):
    texto = str(texto)
    return texto if len(texto) <= max_chars else texto[:max_chars] + "..."


def truncar_labels_ejes(ax, max_chars_x=18, max_chars_y=22, rot_x=None, rot_y=None):
    """
    Aplica truncado a los labels de los ejes X e Y de un Axes de matplotlib.
    - max_chars_x / max_chars_y: longitud máxima de cada label
    - rot_x / rot_y: rotación opcional de las etiquetas
    """
    # EJE X
    x_labels = ax.get_xticklabels()
    if x_labels:
        nuevos_x = [truncar_label(lbl.get_text(), max_chars_x) for lbl in x_labels]
        ax.set_xticklabels(nuevos_x)
        if rot_x is not None:
            ax.tick_params(axis="x", rotation=rot_x)

    # EJE Y
    y_labels = ax.get_yticklabels()
    if y_labels:
        nuevos_y = [truncar_label(lbl.get_text(), max_chars_y) for lbl in y_labels]
        ax.set_yticklabels(nuevos_y)
        if rot_y is not None:
            ax.tick_params(axis="y", rotation=rot_y)
    return ax