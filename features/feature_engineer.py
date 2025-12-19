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
class FeatureEngineering:
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

    def load_data(self):
        print("[INFO] loading data CSVs...")
        self.events = pd.read_csv(self.events_path)
        self.region_baselines = pd.read_csv(self.region_baselines_path)
        self.region_graph = pd.read_csv(self.region_graph_path)
        #ensure types
        self.events['region_id'] = self.events['region_id'].astype(int)
        self.events['is_trip'] = self.events['is_trip'].astype(int)
        print(f"[INFO] events rows: {len(self.events)}")

    def aggregate_region_timeseries(self):
        print("[INFO] aggregating region timeseries...")
        self.events['ts_bin'] = self.events['ts'].dt.floor(self.freq)
        agg = self.events.groupby('region_id', 'ts_bin').agg(
            requests = ('event_id', 'count'),
            trips = ('is_trip', 'sum'),
            avg_price = ('final_price', 'mean'),
            avg_proposed_price = ('proposed_price', 'mean'),
            avg_subsidy = ('subsidy', 'mean'),
            conversion_prob_mean = ('conversion_prob', 'mean'),
            # conversion_rate = ('is_trip', 'sum'),
            avg_context = ('context_score', 'mean'),
            revenue = ('revenue', 'sum')
        )

        agg['conversion_rate'] = agg['trips'] / agg['requests']
        agg['conversion_rate'] = agg['conversion_rate'].fillna(0.0)

        # join supply from region_rates: if not present in events, approximate by region_baselines-driven allocation
        # We'll approximate supply as average per-region supply from Step1 region_rates if not included.
        # If events includes 'supply' per row (it doesn't), you can sum/merge that. 

        # add region baseline information
        agg = agg.merge(self.region_baselines[['region_id','base_price','elasticity']], on='region_id', how='left')

        # reorder and keep important columns
        agg = agg[['ts','region_id','requests','trips','conversion_rate','avg_price','avg_proposed_price','avg_subsidy','avg_context','conversion_prob_mean','revenue','base_price','elasticity']]
        self.region_ts = agg.sort_values(['region_id','ts']).reset_index(drop=True)
        print("[INFO] aggregated region_ts shape:", self.region_ts.shape)
        return self.region_ts
    
    