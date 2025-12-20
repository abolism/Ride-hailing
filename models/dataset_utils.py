# models/dataset_utils.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    """
    Creates sliding-window sequences per region.
    Expects dataframe with columns: ts, region_id, and feature cols + target_col
    Each sample X is a sequence of length seq_len of feature vectors, and y is the scalar target at t+horizon.
    """
    def __init__(self, df: pd.DataFrame, feature_cols, target_col, seq_len=12, horizon=1):
        super().__init__()
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        # group by region and build index of windows
        self.windows = []  # list of tuples (region_id, start_idx)
        self.region_maps = {}  # region_id -> dataframe slice (as numpy array)
        # ensure sorted
        self.df = self.df.sort_values(['region_id','ts']).reset_index(drop=True)

        for region, g in self.df.groupby('region_id'):
            g_sorted = g.sort_values('ts').reset_index(drop=True)
            feats = g_sorted[self.feature_cols].to_numpy(dtype=float)
            targets = g_sorted[self.target_col].to_numpy(dtype=float)
            n = len(g_sorted)
            # for each start index i, we need indices i .. i+seq_len-1 for X and target at i+seq_len-1+horizon
            max_start = n - (self.seq_len + self.horizon) + 1
            if max_start <= 0:
                continue
            base_idx = len(self.windows)
            for i in range(max_start):
                # store (region, idx_of_start_in_global_df) but simpler store region and start integer relative to region
                self.windows.append((region, i))
            # store arrays for region
            self.region_maps[region] = {
                'feats': feats,
                'targets': targets,
                'n': n
            }

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        region, start = self.windows[idx]
        entry = self.region_maps[region]
        feats = entry['feats']
        targets = entry['targets']
        seq = feats[start : start + self.seq_len]  # shape (seq_len, n_features)
        y = targets[start + self.seq_len + self.horizon - 1]
        # convert to tensors
        x_t = torch.tensor(seq, dtype=torch.float32)
        y_t = torch.tensor(float(y), dtype=torch.float32)
        return x_t, y_t

def build_feature_matrix_from_region_agg(df, features_to_use=None, target_col='requests_forward'):
    """
    Accepts the forecast dataset with column 'ts', 'region_id', feature cols and target_col.
    If target_col missing, compute target as groupby shift.
    """
    df = df.copy()
    # ensure ts is datetime
    df['ts'] = pd.to_datetime(df['ts'])
    # if target not present compute from requests & horizon inference not available, so raise
    if target_col not in df.columns:
        raise RuntimeError(f"target column '{target_col}' missing in dataset. Ensure forecast_dataset.csv contains label 'requests_forward'.")
    # determine feature columns if not provided
    if features_to_use is None:
        exclude = {'ts','region_id', target_col}
        features_to_use = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return df, features_to_use

def train_val_test_split_timewise(df, val_ratio=0.15, test_ratio=0.15):
    """
    Splits by time per region: computes indexes for train/val/test by timestamps.
    Returns three dataframes (train_df, val_df, test_df).
    Strategy: for each region, split its time sorted indices accordingly and concat.
    """
    dfs = []
    train_list, val_list, test_list = [], [], []
    for region, g in df.groupby('region_id'):
        g_sorted = g.sort_values('ts').reset_index(drop=True)
        n = len(g_sorted)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = max(0, n - n_val - n_test)
        # handle tiny series
        if n_train <= 0:
            # fallback: allocate 60/20/20 if very small
            n_train = max(1, int(n * 0.6))
            n_val = max(0, int(n * 0.2))
            n_test = n - n_train - n_val
            if n_test < 0:
                n_test = 0
        train_list.append(g_sorted.iloc[:n_train])
        val_list.append(g_sorted.iloc[n_train:n_train + n_val])
        test_list.append(g_sorted.iloc[n_train + n_val:])
    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    return train_df, val_df, test_df
