import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from collections import Counter


class CDFDataset(Dataset):
    def __init__(self, filepath="query_workload.csv", num_items=500000, distribution='zipf'):
        self.items = []

        print("\n[DATASET] Initializing Workload CDF Dataset...")

        if os.path.exists(filepath):
            print(f"[INFO] Loading queries from '{filepath}'")
            try:
                df = pd.read_csv(filepath)
                df.columns = df.columns.str.strip()
                if "Key" in df.columns:
                    self.items = df["Key"].dropna().astype(int).tolist()
            except Exception as e:
                print(f"[ERROR] Failed reading CSV: {e}.")

        # Fallback to generating a synthetic query workload
        if not self.items:
            print(f"[INFO] Generating {num_items:,} synthetic queries (Distribution: {distribution})")
            if distribution == 'zipf':
                s = np.random.zipf(a=1.5, size=num_items)
                self.items = [int(x) for x in s if x < 1000000]
            else:
                self.items = np.random.randint(0, 100000, num_items).tolist()

        sorted_items = sorted(self.items)
        n = len(sorted_items)
        self.unique_keys = sorted(list(set(sorted_items)))

        self.min_key = self.unique_keys[0] if self.unique_keys else 0
        self.max_key = self.unique_keys[-1] if self.unique_keys else 0

        self.label_map = {}
        counts = Counter(sorted_items)
        cumulative_mass = 0.0

        for k in self.unique_keys:
            freq = counts[k]
            cumulative_mass += freq
            self.label_map[k] = cumulative_mass / n

        print("-" * 40)
        print(f"[SUCCESS] Dataset Ready!")
        print(f"          -> Unique Queried Keys: {len(self.unique_keys):,}")
        print(f"          -> Key Range:   [{self.min_key}, {self.max_key}]")
        print("-" * 40 + "\n")

    def __len__(self):
        return len(self.unique_keys)

    def __getitem__(self, idx):
        key = self.unique_keys[idx]
        label = self.label_map[key]

        # Convert Integer -> Normalized Float [0.0, 1.0]
        norm_key = key / self.max_key if self.max_key > 0 else 0.0

        # A single continuous feature
        features = torch.tensor([norm_key], dtype=torch.float32)

        return features, torch.tensor(label, dtype=torch.float32)

    def get_cdf(self, key):
        return self.label_map.get(key, 0.0)
