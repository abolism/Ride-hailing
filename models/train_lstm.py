# models/train_lstm.py
import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from lstm_model import LSTMForecastModel
from dataset_utils import TimeSeriesDataset, build_feature_matrix_from_region_agg, train_val_test_split_timewise

# metrics
def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))

def make_dataloaders(train_df, val_df, test_df, feature_cols, target_col, seq_len, horizon, batch_size):
    train_ds = TimeSeriesDataset(train_df, feature_cols, target_col, seq_len=seq_len, horizon=horizon)
    val_ds   = TimeSeriesDataset(val_df, feature_cols, target_col, seq_len=seq_len, horizon=horizon)
    test_ds  = TimeSeriesDataset(test_df, feature_cols, target_col, seq_len=seq_len, horizon=horizon)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X, y in loader:
        X = X.to(device); y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        n += X.size(0)
    return total_loss / max(n,1)

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device); y = y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item() * X.size(0)
            preds.append(y_hat.cpu().numpy())
            trues.append(y.cpu().numpy())
            n += X.size(0)
    if n == 0:
        return None, None, None
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return total_loss / n, mae(preds, trues), rmse(preds, trues)

def main(args):
    # read dataset
    df = pd.read_csv(args.dataset, parse_dates=['ts'])
    df, feature_cols = build_feature_matrix_from_region_agg(df, features_to_use=None, target_col=args.target_col)
    print("[INFO] dataset shape:", df.shape)
    # train/val/test split
    train_df, val_df, test_df = train_val_test_split_timewise(df, val_ratio=args.val_frac, test_ratio=args.test_frac)
    print(f"[INFO] train/val/test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # feature scaling: fit scaler on train set (using only feature cols)
    scaler = StandardScaler()
    # fit on train feature rows (grouped order doesn't matter)
    scaler.fit(train_df[feature_cols].fillna(0.0).to_numpy())
    # apply transform to all splits (and store as numpy arrays inside the DataFrames for dataset to read)
    def apply_scaler_to_df(d):
        if len(d)==0:
            return d
        arr = scaler.transform(d[feature_cols].fillna(0.0).to_numpy())
        for i, c in enumerate(feature_cols):
            d[c] = arr[:, i]
        return d

    train_df = apply_scaler_to_df(train_df)
    val_df = apply_scaler_to_df(val_df)
    test_df = apply_scaler_to_df(test_df)

    # make dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(
        train_df, val_df, test_df, feature_cols, args.target_col, seq_len=args.seq_len, horizon=args.horizon, batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print("[INFO] using device:", device)

    model = LSTMForecastModel(input_size=len(feature_cols), hidden_size=args.hidden_size, num_layers=args.num_layers,
                              dropout=args.dropout, bidirectional=args.bidirectional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = eval_epoch(model, val_loader, loss_fn, device)
        if val_metrics[0] is None:
            print("[WARN] validation set empty or no sequences available. Skipping validation.")
            val_loss, val_mae, val_rmse = None, None, None
        else:
            val_loss, val_mae, val_rmse = val_metrics
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}")

        # checkpoint by val_rmse
        if val_rmse is not None and val_rmse < best_val_rmse - 1e-6:
            best_val_rmse = val_rmse
            best_state = model.state_dict()
            no_improve = 0
            print("[INFO] best model updated (val_rmse)", best_val_rmse)
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print("[INFO] early stopping triggered.")
            break

    # save best model & scaler
    model_path = args.model_path
    scaler_path = args.scaler_path
    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump({'scaler': scaler, 'feature_cols': feature_cols}, f)
    print("[DONE] saved model:", model_path)
    print("[DONE] saved scaler:", scaler_path)

    # final test eval
    test_metrics = eval_epoch(model, test_loader, loss_fn, device)
    if test_metrics[0] is not None:
        test_loss, test_mae, test_rmse = test_metrics
        print(f"[TEST] loss={test_loss:.6f} mae={test_mae:.4f} rmse={test_rmse:.4f}")
    else:
        print("[WARN] no test sequences to evaluate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/features/forecast_dataset.csv")
    parser.add_argument("--target_col", default="requests_forward")
    parser.add_argument("--seq_len", type=int, default=12)   # e.g., 12*5min=1 hour
    parser.add_argument("--horizon", type=int, default=12)   # predict 12 steps ahead if dataset uses 5min steps
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--model_path", default="models/lstm_model.pt")
    parser.add_argument("--scaler_path", default="models/scaler.pkl")
    args = parser.parse_args()
    main(args)
