import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse

# Utility Functions
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def price_to_conversion_prob(price, base_price, elasticity, context_score):
    '''
    map price and context to conversion probability
    Model: log-odds = alpha - beta * (price/base_price - 1) + context_score
    '''

    alpha = 0.5
    beta = max(1e-6, abs(elasticity)) * 3.0
    logit = alpha - beta * (price - base_price) / (base_price + 1e-9) + context_score
    return sigmoid(logit)

def apply_subsidy(price, subsidy):
    min_price = 0.1
    return max(min_price, price - subsidy)

class DataSimulator:
    def __init__(self, 
                 n_regions=10, 
                 n_riders=500, 
                 n_drivers=200, 
                 start="2025-12-01T00:00:00",
                 end="2025-12-02T00:00:00",
                 freq="5min",
                 seed=42):
        self.n_riders = n_riders
        self.n_regions = n_regions
        self.n_drivers = n_drivers
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.freq = freq
        self.rng = np.random.default_rng(seed)
        self.region_baselines = None
        self.region_graph = None
        self.time_index = None
        self.region_rates = None
        self.events = None

    def generate_region_baselines(self):
        "Generate per-region base demand rates, base price, elasticity"
        rows = []
        for r in range(self.n_regions):
            base_rate = float(self.rng.uniform(0.2, 3.0))
            base_price = float(self.rng.uniform(1.0, 3.0))
            elasticity = float(self.rng.uniform(-2.0, -1.5))
            rows. append({
                "region_id": int(r),
                "base_rate": base_rate,
                "base_price": base_price,
                "elasticity": elasticity
            })

        self.region_baselines = pd.DataFrame(rows)
        return self.region_baselines
    
    def generate_region_graph(self, k_neighbors=2, edge_prob=0.3):
        seed_val = int(self.rng.integers(1<<30))
        G = nx.watts_strogatz_graph(n=self.n_regions, k=k_neighbors, p=edge_prob, seed=seed_val)
        edges = []
        for a,b in G.edges():
            w = float(self.rng.uniform(0.1,1.0))
            edges.append({
                "region_a": a, "region_b": b, "weight": w
            })

        self.region_graph = edges
        return self.region_graph
    def generate_region_graph_variable_k(self, mean_k=2.0, edge_prob=0.3,
                                     degree_dist='poisson', pop_attractiveness=None,
                                     min_k=1, ensure_connected=True):
        """
        Variant of Watts-Strogatz that allows variable k per node.
        - Uses self.rng for all randomness (keeps reproducibility from seed).
        - mean_k: average k if using poisson or normal.
        - degree_dist: 'poisson', 'normal', 'proportional', or 'powerlaw'
        - pop_attractiveness: optional array-like of length n_regions to scale k_i
        - min_k: minimum neighbors per node
        """
        n = int(self.n_regions)
        rng = self.rng

        # --- 1) sample desired degree k_i for each node ---
        if degree_dist == 'poisson':
            # sample Poisson and enforce at least min_k, at most n-1
            k_list = rng.poisson(lam=mean_k, size=n).astype(int)
        elif degree_dist == 'normal':
            k_list = np.round(rng.normal(loc=mean_k, scale=max(1.0, mean_k/2), size=n)).astype(int)
        elif degree_dist == 'proportional' and pop_attractiveness is not None:
            pop = np.asarray(pop_attractiveness)
            # scale pop to have mean mean_k
            pop = np.maximum(pop, 1e-6)
            k_list = np.round(pop / pop.mean() * mean_k).astype(int)
        elif degree_dist == 'powerlaw':
            # draw from a Pareto-like heavy tail then scale
            raw = rng.pareto(a=2.5, size=n)  # tunable alpha
            k_list = np.round(raw / raw.mean() * mean_k).astype(int)
        else:
            # fallback: constant k
            k_list = np.full(n, int(max(min_k, round(mean_k))), dtype=int)

        # clamp and ensure at least min_k and at most n-1
        k_list = np.clip(k_list, min_k, n-1)

        # For ring-connection convenience we'll interpret k_i as total neighbors
        # and connect floor(k_i/2) clockwise and floor(k_i/2) counterclockwise, extra goes clockwise.
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # --- 2) connect ring-local neighbors according to k_list ---
        for i in range(n):
            k = int(k_list[i])
            half = k // 2
            extra = k - 2 * half  # 0 or 1
            for step in range(1, half + 1):
                j1 = (i + step) % n
                j2 = (i - step) % n
                G.add_edge(i, j1)
                G.add_edge(i, j2)
            if extra == 1:
                # connect the extra clockwise neighbor
                j = (i + half + 1) % n
                G.add_edge(i, j)

        # --- 3) WS-style rewiring of the *original* ring-local edges ---
        # To avoid double-rewiring, consider each edge where (i < j) and j is the clockwise neighbor used originally.
        # We'll make a copy of the original edges list to decide rewiring.
        original_edges = list(G.edges())
        for (i, j) in original_edges:
            # decide rewiring with probability edge_prob
            if rng.random() < edge_prob:
                # remove this edge and choose a new target for i
                if G.has_edge(i, j):
                    G.remove_edge(i, j)
                # pick new node r not equal to i, and avoid creating parallel edge
                attempts = 0
                while True:
                    r = int(rng.integers(0, n))
                    attempts += 1
                    if r == i: 
                        continue
                    if not G.has_edge(i, r):
                        G.add_edge(i, r)
                        break
                    # safety: if it's getting stuck, break and skip rewiring
                    if attempts > 10 * n:
                        break

        # --- 4) optionally ensure the graph is connected ---
        if ensure_connected:
            if not nx.is_connected(G):
                # connect components by adding edges between representative nodes
                comps = list(nx.connected_components(G))
                for a, b in zip(comps[:-1], comps[1:]):
                    node_a = next(iter(a))
                    node_b = next(iter(b))
                    G.add_edge(node_a, node_b)

        # --- 5) assign weights using same RNG ---
        edges = []
        for a, b in G.edges():
            w = float(rng.uniform(0.1, 1.0))
            edges.append({"region_a": int(a), "region_b": int(b), "weight": w})

        self.region_graph = pd.DataFrame(edges)
        return self.region_graph, k_list  # returning k_list is useful for diagnostics
    
    # def simulate_time_index(self):
    #     self.time_index = pd.date_range(start=self.start, end=self.end, freq=self.freq, closed='left')
    #     return self.time_index

    def simulate_time_index(self):
        """
        Create a DatetimeIndex from start (inclusive) to end (exclusive) with frequency self.freq.
        Some pandas versions don't support the 'closed' argument, so we create the full range
        and then remove any timestamps >= self.end to emulate closed='left'.
        """
        # generate candidate range (may include the end)
        idx = pd.date_range(start=self.start, end=self.end, freq=self.freq)

        # keep only timestamps strictly less than end to emulate closed='left'
        idx = idx[idx < self.end]

        # if idx is empty for some corner cases, ensure at least the start is present
        if len(idx) == 0:
            idx = pd.DatetimeIndex([self.start])

        self.time_index = idx
        return self.time_index
    
    def simulate_region_level_rates(self):
        if self.region_baselines is None:
            self.generate_region_baselines()

        if self.time_index is None:
            self.simulate_time_index()

        rows = []
        hours = np.array([t.hour for t in self.time_index])
        # simple diurnal pattern: peak at 8-9 and 17-19
        hour_multiplier = 1.0 + 1.2 * np.exp(-0.5 * ((hours - 8)/2.5**2)) + 1.5 * np.exp(-0.5 * ((hours - 18)/2.5**2))
        hour_multiplier = hour_multiplier / hour_multiplier.mean()

        dow = np.array([t.dayofweek for t in self.time_index])
        dow_multiplier = 1.0 + 0.2 * ((dow > 5).astype(float))

        for i, ts in enumerate(self.time_index):
            for _, row in self.region_baselines.iterrows():
                rid = int(row['region_id'])
                base_rate = float(row['base_rate'])
                lam = base_rate * float(hour_multiplier[i]) * float(dow_multiplier[i])
                # add small noise
                lam = max(0.01, lam * float(1.0 + self.rng.normal(0.0, 0.05)))
                # supply drivers per region roughly proportional to drivers distribution
                # allocate drivers randomly but stable
                rows.append({
                    "ts": ts,
                    "region_id": rid,
                    "rate": lam,
                    "supply": int(max(1, np.round(self.n_drivers * (1.0/self.n_regions) * (1.0 + self.rng.normal(0,0.2)))))
                })

        self.region_rates = pd.DataFrame(rows)
        return self.region_rates
    
    def simulate_events(self, subsidy_arms=[0.0, 0.5, 1.0], max_requests=None):
        """
        Simulate per-request attempts sampled from Poisson(rate) at each (ts, region).
        For each request:
         - assign rider_id randomly
         - sample offered price = base_price * (1 + surge)  (surge depends on supply/demand)
         - choose a subsidy arm (we'll sample randomly initially) - downstream experiments will change this
         - compute conversion probability via price_to_conversion_prob()
         - sample is_trip ~ Bernoulli(prob)
         - if many requests > supply, some are dropped/unmatched (supply constraint)
        Returns DataFrame events
        """

        if self.region_rates is None:
            self.simulate_region_level_rates()
        if self.region_baselines is None:
            self.generate_region_baselines()

        events = []
        eid = 0

        base_price_map = dict(self.region_baselines.set_index('region_id')['base_price'])
        elasticity_map = dict(self.region_baselines.set_index('region_id')['elasticity'])

        for _, rr in self.region_rates.iterrows():
            ts = rr['ts']
            region = rr['region_id']
            lam = rr['rate']
            supply = rr['supply']

            n_req = int(self.rng.poisson(lam))

            if max_requests is not None and n_req > max_requests:
                n_req = max_requests
            if n_req == 0:
                continue

            # surge factor: simple function of demand/supply ratio
            # expected demand in this window = lam; current supply = supply
            # surge_multiplier = 1 + gamma * max(0, (demand - supply)/supply)

            gamma = 0.5
            surge_mul = 1 + gamma * max(0, (lam - supply) / (supply + 1e-9))

            # for each request
            for _ in range(n_req):
                eid += 1
                rider_id = int(self.rng.integers(0, self.n_riders))
                # assign a driver later when is_trip is true
                driver_id = -1 # placeholder for when we match later
                base_price = base_price_map[region]
                proposed_price = float(base_price * surge_mul)
                # choose a subsidy arm uniformly
                subsidy = float(self.rng.choice(subsidy_arms, p=None))
                final_price = apply_subsidy(proposed_price, subsidy)
                # context score: a small vector for rider propensity and time effect
                rider_propensity = float(self.rng.normal(0, 0.5))
                context_score = rider_propensity + self.rng.normal(0, 0.1)
                conversion_prob = price_to_conversion_prob(final_price, base_price, elasticity_map[region], context_score)
                is_trip = int(self.rng.random() < conversion_prob)
                revenue = final_price if is_trip else 0.0
                events.append({
                    "event_id": eid,
                    "ts": ts,
                    "region_id": region,
                    "rider_id": rider_id,
                    "driver_id": driver_id,
                    "base_price": round(base_price, 3),
                    "surge_multiplier": round(float(surge_mul), 4),
                    "proposed_price": round(proposed_price, 3),
                    "subsidy": round(subsidy, 3),
                    "final_price": round(final_price, 3),
                    "context_score": round(context_score, 4),
                    "conversion_prob": round(float(conversion_prob), 6),
                    "is_trip": is_trip,
                    "revenue": round(float(revenue), 3)
                })

        events_df = pd.DataFrame(events)

        if not events_df.empty:
            events_df.sort_values(['ts', 'region_id'], inplace=True)
            # supply constraints per (ts, region)
            def assign_drivers(group):
                supply = int(group.iloc[0]['region_id'])  # placeholder -> we'll map supply from region_rates
                # real supply from region_rates:
                ts = group.name[0]
                region = group.name[1]
                supply_row = self.region_rates[(self.region_rates.ts == ts) & (self.region_rates.region_id == region)]
                supply = int(supply_row['supply'].iloc[0]) if not supply_row.empty else 1
                # find trip requests
                trip_idx = group[group.is_trip == 1].index.tolist()
                # cap by supply
                assigned = trip_idx[:supply]
                # assign driver ids randomly from pool allocated for region
                drivers_pool = list(range(region * 1000, region * 1000 + supply))
                for i, idx in enumerate(assigned):
                    group.at[idx, 'driver_id'] = drivers_pool[i % max(1,len(drivers_pool))]
                # for trip requests beyond supply -> mark is_trip to 0 (unmatched)
                for idx in trip_idx[supply:]:
                    group.at[idx, 'is_trip'] = 0
                    group.at[idx, 'driver_id'] = -1
                    group.at[idx, 'revenue'] = 0.0
                return group

            # group by ts & region and assign
            events_df = events_df.groupby(['ts', 'region_id'], group_keys=False).apply(assign_drivers).reset_index(drop=True)

        # store
        self.events = events_df
        return self.events
    
    # Save helpers
    def save_events_csv(self, path):
        if self.events is None:
            raise RuntimeError("No events generated yet. Call simulate_events() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.events.to_csv(path, index=False)
        print(f"[INFO] saved events to {path}")

    # def save_region_graph(self, path):
    #     if self.region_graph is None:
    #         raise RuntimeError("No region graph. Call generate_region_graph() first.")
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     self.region_graph.to_csv(path, index=False)
    #     print(f"[INFO] saved region graph to {path}")

    def save_region_graph(self, path):
        """
        Save region graph to CSV. Accepts either a DataFrame or a list-of-dicts
        (defensive: converts list -> DataFrame).
        """
        if self.region_graph is None:
            raise RuntimeError("No region graph. Call generate_region_graph() first.")
        # Accept list-of-dicts or DataFrame
        rg = self.region_graph
        if isinstance(rg, list):
            try:
                rg = pd.DataFrame(rg)
            except Exception as e:
                raise RuntimeError("region_graph is a list but could not convert to DataFrame") from e
        elif not isinstance(rg, pd.DataFrame):
            # try to coerce into DataFrame for safety
            rg = pd.DataFrame(rg)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        rg.to_csv(path, index=False)
        print(f"[INFO] saved region graph to {path}")
        # also keep the normalized DataFrame on the object for later use
        self.region_graph = rg

    def save_region_baselines(self, path):
        if self.region_baselines is None:
            raise RuntimeError("No region baselines. Call generate_region_baselines() first.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.region_baselines.to_csv(path, index=False)
        print(f"[INFO] saved region baselines to {path}")

    
# ---------------------
# CLI
# ---------------------
def main(args):
    sim = DataSimulator(
        n_regions=args.n_regions,
        n_riders=args.n_riders,
        n_drivers=args.n_drivers,
        start=args.start,
        end=args.end,
        freq=args.freq,
        seed=args.seed
    )
    print("[INFO] generating region baselines...")
    sim.generate_region_baselines()
    sim.save_region_baselines(os.path.join(args.out_dir, "region_baselines.csv"))

    print("[INFO] generating region graph...")
    sim.generate_region_graph(k_neighbors=args.k_neighbors, edge_prob=args.edge_prob)
    sim.save_region_graph(os.path.join(args.out_dir, "region_graph.csv"))

    print("[INFO] generating time index & region-level rates...")
    sim.simulate_time_index()
    sim.simulate_region_level_rates()

    print("[INFO] simulating events...")
    sim.simulate_events(subsidy_arms=args.subsidy_arms, max_requests=args.max_requests)
    events_path = os.path.join(args.out_dir, "events.csv")
    sim.save_events_csv(events_path)
    print("[DONE] data generation complete. events:", sim.events.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data", help="output directory")
    parser.add_argument("--n_regions", type=int, default=10)
    parser.add_argument("--n_riders", type=int, default=500)
    parser.add_argument("--n_drivers", type=int, default=200)
    parser.add_argument("--start", type=str, default="2025-12-01T00:00:00")
    parser.add_argument("--end", type=str, default="2025-12-02T00:00:00")
    parser.add_argument("--freq", type=str, default="5min")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_neighbors", type=int, default=2)
    parser.add_argument("--edge_prob", type=float, default=0.3)
    parser.add_argument("--subsidy_arms", nargs='+', type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--max_requests", type=int, default=None, help="cap per (ts,region)")
    args = parser.parse_args()
    main(args)
