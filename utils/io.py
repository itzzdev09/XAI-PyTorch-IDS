import os
import pandas as pd

def load_selected_csvs(folder_path, selected_files=None):
    """
    Load multiple CSV files from a folder. 
    If selected_files is None, load all CSVs in the folder.
    """
    if selected_files is None:
        selected_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    dfs = []
    for file in selected_files:
        path = os.path.join(folder_path, file)
        dfs.append(pd.read_csv(path))

    return pd.concat(dfs, ignore_index=True)
