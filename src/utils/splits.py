import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def patient_level_split(df: pd.DataFrame, group_col: str, test_frac: float, seed: int, stratify_col: str):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx = np.arange(len(df))
    groups = df[group_col].values
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
