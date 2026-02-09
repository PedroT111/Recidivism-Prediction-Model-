import pandas as pd

CSV_PATH = "data/df_target.csv"
def load_data() -> pd.DataFrame:
    """
    Load the preprocessed dataset used in the notebook.
    Make sure df_target.csv is located in the data/ folder.
    """
    return pd.read_csv(CSV_PATH)