import numpy as np
import pandas as pd
import argparse
import os


def generate_synthetic_log(input_data, output_train, output_test, num_queries, zipf_a):
    print(f"[INFO] Loading dataset from '{input_data}'...")
    if not os.path.exists(input_data):
        print(f"[ERROR] Could not find {input_data}. Run download_datasets.py first.")
        return

    df_data = pd.read_csv(input_data)

    # Extract unique keys ranked by frequency
    value_counts = df_data['Key'].value_counts()
    unique_keys = value_counts.index.tolist()
    num_unique = len(unique_keys)

    print(f"[INFO] Found {num_unique:,} unique keys.")
    print(f"--- Generating SYNTHETIC TRAIN & TEST queries (Zipf a={zipf_a}) ---")

    def create_queries(size):
        queries = []
        while len(queries) < size:
            batch = np.random.zipf(a=zipf_a, size=size)
            valid_ranks = batch[batch <= num_unique]
            mapped_keys = [unique_keys[rank - 1] for rank in valid_ranks]
            queries.extend(mapped_keys)
        return queries[:size]

    train_queries = create_queries(num_queries)
    test_queries = create_queries(num_queries)

    pd.DataFrame({'Key': train_queries}).to_csv(output_train, index=False)
    pd.DataFrame({'Key': test_queries}).to_csv(output_test, index=False)
    print(f"[SUCCESS] Saved synthetic query logs to '{output_train}' and '{output_test}'")


def generate_sequential_log(input_data, output_train, output_test):
    print(f"[INFO] Loading sequential queries from '{input_data}'...")
    if not os.path.exists(input_data):
        print(f"[ERROR] Could not find {input_data}. Run download_datasets.py first.")
        return

    df = pd.read_csv(input_data)
    queries = df['Key'].tolist()
    total_queries = len(queries)

    print(f"[INFO] Loaded {total_queries:,} sequential queries.")
    print(f"--- Splitting CHRONOLOGICALLY to preserve temporal locality ---")

    # 80/20 Chronological split to prevent data leakage but maintain time-series properties
    split_idx = int(total_queries * 0.8)
    train_queries = queries[:split_idx]
    test_queries = queries[split_idx:]

    pd.DataFrame({'Key': train_queries}).to_csv(output_train, index=False)
    pd.DataFrame({'Key': test_queries}).to_csv(output_test, index=False)
    print(f"[SUCCESS] Train: {len(train_queries):,} queries | Test: {len(test_queries):,} queries")
    print(f"[SUCCESS] Saved sequential query logs to '{output_train}' and '{output_test}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Query Log")
    parser.add_argument('--input', type=str, required=True, help="Downloaded dataset CSV (e.g., bbc_queries.csv)")
    parser.add_argument('--output_train', type=str, default='query_workload_train.csv', help="Output train log CSV")
    parser.add_argument('--output_test', type=str, default='query_workload_test.csv', help="Output test log CSV")
    parser.add_argument('--mode', type=str, choices=['synthetic', 'sequential'], required=True, help="Generation mode")

    parser.add_argument('--queries', type=int, default=500000, help="Total queries per file (synthetic only)")
    parser.add_argument('--zipf', type=float, default=1.5, help="Zipfian skew parameter (synthetic only)")

    args = parser.parse_args()

    if args.mode == 'synthetic':
        generate_synthetic_log(args.input, args.output_train, args.output_test, args.queries, args.zipf)
    elif args.mode == 'sequential':
        generate_sequential_log(args.input, args.output_train, args.output_test)
