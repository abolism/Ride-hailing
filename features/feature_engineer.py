"""
features/feature_engineer.py

Usage (example):
python features/feature_engineer.py \
  --events data/events.csv \
  --region_baselines data/region_baselines.csv \
  --region_graph data/region_graph.csv \
  --freq 5min \
  --out_dir data/features \
  --horizon 12

Outputs:
 - region_agg.csv  (region x ts aggregated)
 - rider_segments.csv
 - forecast_dataset.csv
 - bandit_dataset.csv
"""
import os
import pandas as pd
import argparse
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# Helper functions
# -------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def parse_freq_to_minutes(freq):
    if freq.endswith('min'):
        return int(freq[-3])
    if freq.endswith('H') or freq.endswith('h'):
        return int(freq[-1]) * 60
    raise ValueError("Unsopported freq format.")


# -------------------------
# FeatureEngineer class
# -------------------------
class FeatureEngineer:
    def __init__(self, events_path, region_baselines_path, region_graph_path, freq="5min"):
        self.events_path = events_path
        self.region_baselines_path = region_baselines_path
        self.region_graph_path = region_graph_path
        self.freq = freq
        self.events = None
        self.region_baselines = None
        self.region_graph = None
        self.region_ts = None
        self.rider_segments = None

    # def load_data(self):
    #     print("[INFO] loading data CSVs...")
    #     self.events = pd.read_csv(self.events_path)
    #     self.region_baselines = pd.read_csv(self.region_baselines_path)
    #     self.region_graph = pd.read_csv(self.region_graph_path)
    #     #ensure types
    #     self.events['region_id'] = self.events['region_id'].astype(int)
    #     self.events['is_trip'] = self.events['is_trip'].astype(int)
    #     print(f"[INFO] events rows: {len(self.events)}")

    def load_data(self):
        """
        Robust loading of CSVs. Ensures ts is datetime and drops / reports malformed rows.
        """
        print("[INFO] loading data CSVs...")
        # try to read and parse dates; fall back if needed
        try:
            self.events = pd.read_csv(self.events_path, parse_dates=["ts"])
        except Exception:
            self.events = pd.read_csv(self.events_path)

        # force-convert ts -> datetime, coerce errors to NaT (no infer_datetime_format)
        self.events['ts'] = pd.to_datetime(self.events['ts'], errors='coerce')

        # report and drop malformed timestamp rows
        n_bad = int(self.events['ts'].isna().sum())
        if n_bad > 0:
            print(f"[WARN] {n_bad} rows have invalid timestamps and will be dropped. Showing up to 5 examples:")
            print(self.events[self.events['ts'].isna()].head(5).to_string(index=False))
            self.events = self.events[~self.events['ts'].isna()].copy()

        # normalize types for important columns if present
        if 'region_id' in self.events.columns:
            try:
                self.events['region_id'] = self.events['region_id'].astype(int)
            except Exception:
                self.events['region_id'] = pd.to_numeric(self.events['region_id'], errors='coerce').fillna(0).astype(int)

        if 'is_trip' in self.events.columns:
            # ensure integer 0/1
            self.events['is_trip'] = pd.to_numeric(self.events['is_trip'], errors='coerce').fillna(0).astype(int)

        # load region baselines and graph
        self.region_baselines = pd.read_csv(self.region_baselines_path)
        self.region_graph = pd.read_csv(self.region_graph_path)

        print(f"[INFO] events rows: {len(self.events)}")


    # def aggregate_region_timeseries(self):
    #     print("[INFO] aggregating region timeseries...")
    #     self.events['ts_bin'] = self.events['ts'].dt.floor(self.freq)
    #     agg = self.events.groupby(['ts_bin', 'region_id']).agg(
    #         requests = ('event_id', 'count'),
    #         trips = ('is_trip', 'sum'),
    #         avg_price = ('final_price', 'mean'),
    #         avg_proposed_price = ('proposed_price', 'mean'),
    #         avg_subsidy = ('subsidy', 'mean'),
    #         conversion_prob_mean = ('conversion_prob', 'mean'),
    #         # conversion_rate = ('is_trip', 'sum'),
    #         avg_context = ('context_score', 'mean'),
    #         revenue = ('revenue', 'sum')
    #     )

    #     agg['conversion_rate'] = agg['trips'] / agg['requests']
    #     agg['conversion_rate'] = agg['conversion_rate'].fillna(0.0)

    #     # join supply from region_rates: if not present in events, approximate by region_baselines-driven allocation
    #     # We'll approximate supply as average per-region supply from Step1 region_rates if not included.
    #     # If events includes 'supply' per row (it doesn't), you can sum/merge that. 

    #     # add region baseline information
    #     agg = agg.merge(self.region_baselines[['region_id','base_price','elasticity']], on='region_id', how='left')

    #     # reorder and keep important columns
    #     agg = agg[['ts','region_id','requests','trips','conversion_rate','avg_price','avg_proposed_price','avg_subsidy','avg_context','conversion_prob_mean','revenue','base_price','elasticity']]
    #     self.region_ts = agg.sort_values(['region_id','ts']).reset_index(drop=True)
    #     print("[INFO] aggregated region_ts shape:", self.region_ts.shape)
    #     return self.region_ts

    def aggregate_region_timeseries(self) -> pd.DataFrame:
        """
        Aggregate per (ts, region_id) at self.freq frequency.
        Produces columns: ts, region_id, requests, trips, conversion_rate, avg_price, avg_subsidy, avg_proposed_price, revenue, base_price, elasticity
        This version is defensive against index/column naming oddities.
        """
        print("[INFO] aggregating region timeseries...")
        # defensive checks
        if self.events is None:
            raise RuntimeError("events not loaded. Call load_data() first.")
        if 'ts' not in self.events.columns:
            raise RuntimeError("events missing 'ts' column.")
        if 'region_id' not in self.events.columns:
            raise RuntimeError("events missing 'region_id' column.")

        # floor timestamps to freq and create ts_bin
        self.events['ts_bin'] = self.events['ts'].dt.floor(self.freq)

        # perform aggregation
        agg = self.events.groupby(['ts_bin', 'region_id']).agg(
            requests = ('event_id', 'count'),
            trips = ('is_trip', 'sum'),
            avg_price = ('final_price', 'mean'),
            avg_proposed_price = ('proposed_price', 'mean'),
            avg_subsidy = ('subsidy', 'mean'),
            avg_context = ('context_score', 'mean'),
            conversion_prob_mean = ('conversion_prob', 'mean'),
            revenue = ('revenue', 'sum')
        )

        # ensure ts column after reset_index (defensive)
        agg = agg.reset_index()
        # possible column names: 'ts_bin' or 'ts' depending on prior code; normalize to 'ts'
        if 'ts_bin' in agg.columns and 'ts' not in agg.columns:
            agg = agg.rename(columns={'ts_bin': 'ts'})
        elif 'ts' not in agg.columns:
            # try common fallback names
            if 'level_0' in agg.columns:
                agg = agg.rename(columns={'level_0': 'ts'})
            else:
                # create ts column from index if index is datetime-like
                try:
                    if isinstance(agg.index, pd.DatetimeIndex):
                        agg = agg.reset_index().rename(columns={'index':'ts'})
                    else:
                        raise RuntimeError("Unable to find or create 'ts' column in aggregated dataframe.")
                except Exception as e:
                    raise RuntimeError("Failed to normalize aggregated timestamp column: " + str(e))

        # ensure naming & types are correct
        agg['ts'] = pd.to_datetime(agg['ts'])

        # compute conversion_rate
        agg['conversion_rate'] = (agg['trips'] / agg['requests']).fillna(0.0)

        # join region baseline (if available)
        if self.region_baselines is not None and 'region_id' in self.region_baselines.columns:
            # ensure region_baselines types compatible
            self.region_baselines['region_id'] = self.region_baselines['region_id'].astype(int)
            agg = agg.merge(self.region_baselines[['region_id','base_price','elasticity']], on='region_id', how='left')
        else:
            agg['base_price'] = np.nan
            agg['elasticity'] = np.nan

        # Reorder columns defensively: only include columns that actually exist
        desired_cols = ['ts','region_id','requests','trips','conversion_rate','avg_price',
                   'avg_proposed_price','avg_subsidy','avg_context','conversion_prob_mean','revenue','base_price','elasticity']
        existing = [c for c in desired_cols if c in agg.columns]
        agg = agg[existing].copy()

        self.region_ts = agg.sort_values(['region_id','ts']).reset_index(drop=True)
        print("[INFO] aggregated region_ts shape:", self.region_ts.shape)
        return self.region_ts
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df['ts'].dt.hour
        df['dow'] = df['ts'].dt.dayofweek
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        # cyclical encode hour: sin/cos
        df['time_of_day_minutes'] = df['hour'] * 60 + df['ts'].dt.minute
        minutes_in_day = 24 * 60
        df['time_sin'] = np.sin(2 * np.pi * df['time_of_day_minutes'] / minutes_in_day)
        df['time_cos'] = np.cos(2 * np.pi * df['time_of_day_minutes'] / minutes_in_day)
        return df
    
    def create_lag_features(self, df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['region_id','ts'])
        for col in cols:
            for lag in lags:
                new_col = f"{col}_lag_{lag}"
                df[new_col] = df.groupby('region_id')[col].shift(lag)
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, cols: List[str], windows: List[int]) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(['region_id','ts'])
        for col in cols:
            for w in windows:
                df[f"{col}_roll_mean_{w}"] = df.groupby('region_id')[col].transform(lambda x: x.rolling(window=w, min_periods=1).mean())
                df[f"{col}_roll_std_{w}"] = df.groupby('region_id')[col].transform(lambda x: x.rolling(window=w, min_periods=1).std().fillna(0.0))
        return df
    
    def compute_supply_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # approximate supply: if supply not present, assume proportional to base driver allocation
        # create estimated_supply = requests * 0.8 rounded or base driver allocation heuristic
        df['est_supply'] = np.maximum(1.0, (df['requests'].rolling(3, min_periods=1).mean() * 0.6).fillna(1.0))
        df['demand_supply_ratio'] = df['requests'] / (df['est_supply'] + 1e-9)
        df['utilization_est'] = df['trips'] / (df['est_supply'] + 1e-9)
        return df
    
    def simulate_driver_supply(self, total_drivers: int = None, min_per_region: int = 1, random_state: int = 42) -> pd.DataFrame:
        """
        Simulate number of drivers present in each region at each timestamp.
        Returns DataFrame with columns: ts, region_id, supply_raw
        Strategy:
          - For each ts, compute region weights = base_rate * hour_multiplier (same idea as Step1)
          - Normalize weights to probabilities and sample a multinomial(total_drivers)
        """
        # fallback total drivers
        if total_drivers is None:
            # try to infer from object attr (if set) else use sensible default
            total_drivers = getattr(self, 'n_drivers', None) or 200

        # ensure baselines loaded
        if self.region_baselines is None:
            raise RuntimeError("region_baselines is not loaded. Call load_data() first.")

        # ensure region_ts exists or create a sorted unique timestamp list from events
        if self.region_ts is None:
            # aggregate to get ts list (fast)
            self.aggregate_region_timeseries()

        ts_list = sorted(self.region_ts['ts'].unique())

        # compute hour multiplier (same functional form as generation)
        hours = np.array([pd.Timestamp(ts).hour for ts in ts_list])
        hour_multiplier = 1.0 + 1.2 * np.exp(-0.5 * ((hours - 8) / 2.5) ** 2) + 1.5 * np.exp(-0.5 * ((hours - 18) / 2.5) ** 2)
        hour_multiplier = hour_multiplier / hour_multiplier.mean()

        base_rate_map = dict(self.region_baselines.set_index('region_id')['base_rate'])
        region_ids = sorted(self.region_baselines['region_id'].tolist())

        rng = np.random.default_rng(seed=random_state)
        rows = []
        for i, ts in enumerate(ts_list):
            weights = np.array([base_rate_map[r] for r in region_ids], dtype=float)
            weights = weights * float(hour_multiplier[i])
            weights = np.maximum(weights, 1e-9)
            probs = weights / weights.sum()
            alloc = rng.multinomial(total_drivers, probs)
            for idx, r in enumerate(region_ids):
                supply = int(max(min_per_region, alloc[idx]))
                rows.append({'ts': pd.to_datetime(ts), 'region_id': int(r), 'supply_raw': supply})
        supply_df = pd.DataFrame(rows)
        return supply_df


    def compute_supply_demand_features_using_sim(self, df: pd.DataFrame, total_drivers: int = None, min_per_region: int = 1, random_state: int = 42) -> pd.DataFrame:
        """
        Simulate driver supply and merge into aggregated df to compute demand/supply metrics.
        df: aggregated DataFrame with columns ['ts','region_id','requests','trips',...]
        Returns merged DataFrame with added columns: supply, demand_supply_ratio, utilization
        """
        df = df.copy()
        # ensure ts is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ts']):
            df['ts'] = pd.to_datetime(df['ts'])

        supply_df = self.simulate_driver_supply(total_drivers=total_drivers, min_per_region=min_per_region, random_state=random_state)
        # merge on ts & region_id
        merged = df.merge(supply_df, on=['ts', 'region_id'], how='left')
        merged['supply_raw'] = merged['supply_raw'].fillna(min_per_region).astype(int)
        merged['supply'] = merged['supply_raw']
        merged['demand_supply_ratio'] = merged['requests'] / (merged['supply'] + 1e-9)
        merged['utilization'] = merged['trips'] / (merged['supply'] + 1e-9)
        # drop helper column if you want
        merged = merged.drop(columns=['supply_raw'])
        return merged

    def add_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given region_graph edges (region_a, region_b, weight), compute neighbor-weighted avg requests and lagged neighbor requests.
        """
        df = df.copy()
        # build adjacency dict
        adj = {}
        for _, row in self.region_graph.iterrows():
            a, b, w = int(row['region_a']), int(row['region_b']), float(row.get('weight',1.0))
            adj.setdefault(a, []).append((b,w))
            adj.setdefault(b, []).append((a,w))
        # compute neighbor features per ts
        rows = []
        for ts, group in df.groupby('ts'):
            # map region->requests
            rmap = dict(zip(group['region_id'], group['requests']))
            for _, rrow in group.iterrows():
                region = int(rrow['region_id'])
                neigh = adj.get(region, [])
                if len(neigh) == 0:
                    neigh_mean = 0.0
                else:
                    vals = []
                    ws = []
                    for nb, w in neigh:
                        vals.append(float(rmap.get(nb, 0.0)))
                        ws.append(w)
                    if sum(ws) > 0:
                        neigh_mean = np.dot(vals, ws) / (sum(ws) + 1e-9)
                    else:
                        neigh_mean = np.mean(vals) if len(vals)>0 else 0.0
                rows.append({
                    'ts': ts,
                    'region_id': region,
                    'neigh_requests_mean': neigh_mean
                })
        neigh_df = pd.DataFrame(rows)
        df = df.merge(neigh_df, on=['ts','region_id'], how='left')
        df['neigh_requests_mean'] = df['neigh_requests_mean'].fillna(0.0)
        return df


    def cluster_riders(self, n_clusters=8) -> pd.DataFrame:
        """
        Build simple rider segments from events. Use features like avg_subsidy_received, avg_proposed_price, conversion rate per rider.
        Returns and saves rider_segments DataFrame with columns rider_id, segment.
        """
        print("[INFO] clustering riders...")
        # aggregate per rider
        ragg = self.events.groupby('rider_id').agg(
            requests=('event_id','count'),
            trips=('is_trip','sum'),
            avg_subsidy=('subsidy','mean'),
            avg_final_price=('final_price','mean'),
            avg_context=('context_score','mean')
        ).reset_index()
        ragg['conversion_rate'] = ragg['trips'] / (ragg['requests'] + 1e-9)
        # filter low-count riders
        ragg_sample = ragg[ragg['requests'] >= 3].copy()
        feats = ['requests','avg_subsidy','avg_final_price','avg_context','conversion_rate']
        scaler = StandardScaler()
        X = scaler.fit_transform(ragg_sample[feats].fillna(0.0))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        segs = kmeans.fit_predict(X)
        ragg_sample['segment'] = segs
        # assign remaining riders to -1
        ragg = ragg.merge(ragg_sample[['rider_id','segment']], on='rider_id', how='left')
        ragg['segment'] = ragg['segment'].fillna(-1).astype(int)
        self.rider_segments = ragg[['rider_id','segment']]
        print("[INFO] rider_segments shape:", self.rider_segments.shape)
        return self.rider_segments

    def prepare_forecast_dataset(self, horizon=1, lags=[1,2,3,12,24], rolling_windows=[3,12], keep_cols=None) -> pd.DataFrame:
        """
        Prepare a dataset for forecasting next-horizon requests per region.
        label_col: requests shifted by -horizon (we predict t+horizon)
        """
        print("[INFO] preparing forecast dataset...")
        if self.region_ts is None:
            self.aggregate_region_timeseries()
        df = self.region_ts.copy()
        df = self.add_time_features(df)
        # lags on requests and conversion_rate and avg_price
        df = self.create_lag_features(df, cols=['requests','conversion_rate','avg_price'], lags=lags)
        df = self.create_rolling_features(df, cols=['requests','conversion_rate'], windows=rolling_windows)
        # df = self.compute_supply_demand_features(df)
        df = self.compute_supply_demand_features_using_sim(self.region_ts, total_drivers=200, min_per_region=1)
        df = self.add_graph_features(df)
        # label
        df['requests_forward'] = df.groupby('region_id')['requests'].shift(-horizon)
        # drop rows with NaN in label
        df = df.dropna(subset=['requests_forward'])
        # optional keep columns
        if keep_cols is not None:
            df = df[keep_cols]
        return df
    
    def prepare_bandit_dataset(self, lookback=24, context_cols=None) -> pd.DataFrame:
        """
        Prepare event-level data suitable for bandit experiments. For each event (or aggregated user-event),
        include context features and observed reward (is_trip or revenue). The chosen arm is the recorded subsidy.
        We also include rider segment if available.
        """
        print("[INFO] preparing bandit dataset...")
        ev = self.events.copy()
        # merge rider segment if exists
        if self.rider_segments is None:
            self.cluster_riders(n_clusters=8)
        ev = ev.merge(self.rider_segments, on='rider_id', how='left')
        ev['segment'] = ev['segment'].fillna(-1).astype(int)
        # context features: hour, region_id, segment, avg_context (per event already contains context_score)
        ev['hour'] = ev['ts'].dt.hour
        # optionally compute aggregated recent features per rider (lookback windows)
        ev = ev.sort_values(['rider_id','ts'])
        # compute ride_count_last_lookback per rider
        def count_recent(group):
            group = group.copy()
            group['requests_last'] = group['ts'].rolling(f"{lookback}min", on='ts').count() if False else group.groupby('rider_id').cumcount()
            return group
        # simple features: historical conversion rate per rider
        ragg = ev.groupby('rider_id').agg(reqs=('event_id','count'), trips=('is_trip','sum')).reset_index()
        ragg['rider_conv'] = ragg['trips'] / (ragg['reqs'] + 1e-9)
        ev = ev.merge(ragg[['rider_id','rider_conv']], on='rider_id', how='left')
        # context columns
        if context_cols is None:
            context_cols = ['region_id','hour','segment','rider_conv','context_score','avg_proposed_price']
        # rename subsidy as arm
        ev = ev.rename(columns={'subsidy':'arm','is_trip':'reward_binary','revenue':'reward_revenue'})
        # keep relevant cols
        keep = ['event_id','ts','rider_id','region_id','arm','reward_binary','reward_revenue'] + context_cols
        keep = [c for c in keep if c in ev.columns]
        bandit_df = ev[keep].copy()
        return bandit_df

    def save_dataframe(self, df: pd.DataFrame, path: str):
        ensure_dir(os.path.dirname(path))
        df.to_csv(path, index=False)
        print(f"[INFO] saved {path}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--region_baselines", required=True)
    parser.add_argument("--region_graph", required=True)
    parser.add_argument("--freq", default="5min")
    parser.add_argument("--out_dir", default="data/features")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--n_rider_segments", type=int, default=8)
    args = parser.parse_args()

    fe = FeatureEngineer(args.events, args.region_baselines, args.region_graph, freq=args.freq)
    fe.load_data()
    region_ts = fe.aggregate_region_timeseries()
    region_ts = fe.add_time_features(region_ts)
    region_ts = fe.create_lag_features(region_ts, cols=['requests','conversion_rate','avg_price'], lags=[1,2,3,12,24])
    region_ts = fe.create_rolling_features(region_ts, cols=['requests','conversion_rate'], windows=[3,12])
    region_ts = fe.compute_supply_demand_features_using_sim(region_ts, total_drivers=200, min_per_region=1)
    region_ts = fe.add_graph_features(region_ts)
    fe.save_dataframe(region_ts, os.path.join(args.out_dir, "region_agg.csv"))

    rider_segments = fe.cluster_riders(n_clusters=args.n_rider_segments)
    fe.save_dataframe(rider_segments, os.path.join(args.out_dir, "rider_segments.csv"))

    forecast_df = fe.prepare_forecast_dataset(horizon=args.horizon)
    fe.save_dataframe(forecast_df, os.path.join(args.out_dir, "forecast_dataset.csv"))

    bandit_df = fe.prepare_bandit_dataset()
    fe.save_dataframe(bandit_df, os.path.join(args.out_dir, "bandit_dataset.csv"))

    print("[DONE] Feature engineering complete. files written to", args.out_dir)

if __name__ == "__main__":
    main()