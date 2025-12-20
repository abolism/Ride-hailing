# models/infer_lstm.py
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from models.lstm_model import LSTMForecastModel

def load_model_and_scaler(model_path, scaler_path, input_size, device='cpu'):
    m = LSTMForecastModel(input_size=input_size)
    state = torch.load(model_path, map_location=device)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    with open(scaler_path, 'rb') as f:
        meta = pickle.load(f)
    scaler = meta['scaler']
    feature_cols = meta['feature_cols']
    return m, scaler, feature_cols

def predict_from_recent_sequence(model, scaler, feature_cols, recent_df, device='cpu'):
    # recent_df: DataFrame with columns feature_cols and length == seq_len
    arr = recent_df[feature_cols].fillna(0.0).to_numpy()
    arr_scaled = scaler.transform(arr)
    x = torch.tensor(arr_scaled[None, :, :], dtype=torch.float32, device=device)  # (1, seq_len, n_features)
    with torch.no_grad():
        y = model(x).cpu().numpy().ravel()[0]
    return float(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/lstm_model.pt")
    parser.add_argument("--scaler_path", default="models/scaler.pkl")
    parser.add_argument("--recent_csv", required=True, help="CSV with last seq_len rows for a single region, columns must match feature_cols")
    args = parser.parse_args()

    m, scaler, feature_cols = load_model_and_scaler(args.model_path, args.scaler_path, input_size=None, device='cpu')
    recent = pd.read_csv(args.recent_csv)
    pred = predict_from_recent_sequence(m, scaler, feature_cols, recent)
    print("predicted requests:", pred)
