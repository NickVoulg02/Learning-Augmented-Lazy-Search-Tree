import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(ROOT_DIR)

import lazy_tree_original as lst_orig
import lazy_tree_learned as lst_learn

from src.learned.cdf_model import CDFNet

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import random
import time
import csv
import gc

TREE_VARIANTS = ["Set", "Splay", "Treap", "OriginalLazy", "LearnedLazy"]

# Global trackers for CSV writing
GLOBAL_ARGS = None
CURRENT_BENCH_TYPE = "Unknown"


def run_tree_benchmark(tree_name, preload, inserts, queries, mode="batched"):
    if tree_name in ["Set", "Splay", "OriginalLazy"]:
        backend = lst_orig
    else:
        backend = lst_learn

    internal_name = "Lazy" if "Lazy" in tree_name else tree_name

    if mode == "interleaved":
        return backend.run_custom_interleaved(internal_name, preload, inserts, queries)
    elif mode == "interleaved_delete":
        return backend.run_custom_interleaved_delete(internal_name, preload, inserts, queries)
    else:
        return backend.run_custom(internal_name, preload, inserts, queries)


def calculate_max_error(model, loader, device):
    model.eval()
    max_err, total_mae, count = 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds_cdf = model(inputs)
            errors = torch.abs(preds_cdf.squeeze() - targets)
            batch_max = errors.max().item()
            if batch_max > max_err:
                max_err = batch_max
            total_mae += errors.sum().item()
            count += len(targets)
    return max_err, total_mae / count


def train_and_export_oracle(data, hidden_dim=32, epochs=15):
    filename = os.path.abspath(os.path.join(ROOT_DIR, "src", "learned", "weights.txt"))

    data = np.sort(data)
    print(f"\n[LEARN] Training Oracle on {len(data):,} query history items...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(data)
    sample_size = min(n, 25000)
    indices = np.linspace(0, n - 1, sample_size).astype(int)
    sorted_sample = data[indices]
    sampled_labels = indices / n

    max_val = float(sorted_sample[-1]) if len(sorted_sample) > 0 and sorted_sample[-1] > 0 else 1.0
    features = [[float(k) / max_val] for k in sorted_sample]

    X = torch.tensor(features, dtype=torch.float32).to(device)
    y = torch.tensor(sampled_labels, dtype=torch.float32).to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    model = CDFNet(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.MSELoss()

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X).squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        max_err, avg_mae = calculate_max_error(model, val_loader, device)
        scheduler.step(avg_loss)
        print(
            f"        -> Epoch {epoch + 1:02d} | Loss: {avg_loss:.6f} | MaxErr: {max_err:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

    print(f"[SUCCESS] Training finished in {time.time() - start_time:.2f}s")

    with open(filename, 'w') as f:
        f.write(f"MAX_KEY {max_val}\n")
        for name, param in model.named_parameters():
            if 'weight' in name:
                f.write(f"WEIGHT {name}\n")
                weights = param.detach().cpu().numpy()
                if name == 'fc3.weight':
                    weights = weights[0:1, :]
                for row in weights:
                    f.write(" ".join([f"{x:.8f}" for x in row]) + "\n")
            elif 'bias' in name:
                f.write(f"BIAS {name}\n")
                data_param = param.detach().cpu().numpy().flatten()
                if name == 'fc3.bias':
                    data_param = data_param[0:1]
                f.write(" ".join([f"{x:.8f}" for x in data_param]) + "\n")
    print(f"[SUCCESS] Weights exported to {filename}")

    del model, loader, val_loader, X, y, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def print_header(title, params):
    global CURRENT_BENCH_TYPE
    CURRENT_BENCH_TYPE = params.get("Type", title)

    print("\n" + "=" * 60)
    print(f"[BENCHMARK] {title}")
    print("=" * 60)
    for k, v in params.items():
        print(f"  -> {k:<18}: {v}")
    print("-" * 60)


def print_batched_table(results):
    print("\n" + "=" * 145)
    print(
        f"| {'Tree Variant':<14} | {'Ins Time(s)':<11} | {'Qry Time(s)':<11} | {'Total (s)':<12} | {'Ins Comps':<13} | {'Qry Comps':<13} | {'Tot Comps':<13} | {'Mem(MB)':<10} |")
    print("-" * 145)
    for r in results:
        print(
            f"| {r[0]:<14} | {r[1]:<11.4f} | {r[2]:<11.4f} | {r[3]:<12.4f} | {r[4]:<13,} | {r[5]:<13,} | {r[6]:<13,} | {r[8]:<10} |")
    print("=" * 145 + "\n")


def print_interleaved_table(results):
    print("\n" + "=" * 140)
    print(
        f"| {'Tree Variant':<14} | {'Total Time(s)':<13} | {'Throughput':<11} | {'Ins Comps':<13} | {'Qry Comps':<13} | {'Tot Comps':<13} | {'Mem(MB)':<10} |")
    print("-" * 140)
    for r in results:
        print(f"| {r[0]:<14} | {r[1]:<13.4f} | {r[2]:<11,} | {r[4]:<13,} | {r[5]:<13,} | {r[6]:<13,} | {r[7]:<10} |")
    print("=" * 140 + "\n")


def generate_standard_queries(size: int, total_n: int, dist: str, zipf_alpha: float = 1.5,
                              shuffled_keys: list = None) -> list:
    if dist == 'uniform':
        return [random.randint(0, total_n - 1) for _ in range(size)]
    elif dist == 'zipf':
        queries = []
        while len(queries) < size:
            batch = np.asarray(np.random.zipf(a=zipf_alpha, size=size))
            valid_ranks = batch[batch <= total_n]
            if shuffled_keys:
                mapped_keys = [shuffled_keys[rank - 1] for rank in valid_ranks]
                queries.extend(mapped_keys)
            else:
                queries.extend(valid_ranks.tolist())
        return queries[:size]
    return []


def generate_mixed_queries(size: int, total_n: int, zipf_alpha: float, zipf_ratio: float, shuffled_keys: list) -> list:
    num_zipf = int(size * zipf_ratio)
    num_uniform = size - num_zipf

    zipf_queries = []
    while len(zipf_queries) < num_zipf:
        batch = np.asarray(np.random.zipf(a=zipf_alpha, size=num_zipf))
        valid_ranks = batch[batch <= total_n]
        if shuffled_keys:
            mapped_keys = [shuffled_keys[rank - 1] for rank in valid_ranks]
            zipf_queries.extend(mapped_keys)
        else:
            zipf_queries.extend(valid_ranks.tolist())
    zipf_queries = zipf_queries[:num_zipf]

    uniform_queries = [random.randint(0, total_n - 1) for _ in range(num_uniform)]

    mixed_queries = zipf_queries + uniform_queries
    random.shuffle(mixed_queries)

    return mixed_queries


def evaluate_trees(preload_list, insert_list, test_queries, mode="batched", insert_count=0):
    global GLOBAL_ARGS, CURRENT_BENCH_TYPE

    runs = getattr(GLOBAL_ARGS, 'runs', 1)
    dataset = getattr(GLOBAL_ARGS, 'dataset', getattr(GLOBAL_ARGS, 'bench', 'synthetic'))
    ins = getattr(GLOBAL_ARGS, 'insert', 0)
    q = getattr(GLOBAL_ARGS, 'q', 0)

    target_dir = os.path.join(ROOT_DIR, "results_synthetic")
    os.makedirs(target_dir, exist_ok=True)

    csv_file = os.path.join(target_dir, f"results_{dataset}_{ins}ins_{q}qry.csv")
    file_exists = os.path.isfile(csv_file)

    results_avg = []
    baseline = None

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Bench Type", "Tree", "Runs",
                "Insert Time (s)", "Query Time (s)", "Total Time (s)",
                "Insert Comps", "Query Comps", "Total Comps",
                "p99 Latency (s)", "Throughput_QPS", "Memory (MB)", "Status"
            ])

        for tree in TREE_VARIANTS:
            print(f"[PROCESS] Running {tree} (Averaging {runs} runs)...", end='\r')

            totals = {
                "insert_time": 0.0, "query_time": 0.0, "total_time": 0.0,
                "insert_comps": 0, "query_comps": 0, "total_comps": 0,
                "p99": 0.0, "throughput": 0, "qps": 0, "memory": 0.0
            }
            status = "OK"

            for _ in range(runs):
                res = run_tree_benchmark(tree, preload_list, insert_list, test_queries, mode)

                if baseline is None:
                    baseline = res.checksum
                    status = "BASE"
                elif res.checksum != baseline and status != "BASE":
                    status = "FAIL"

                totals["insert_time"] += res.insert_time
                totals["query_time"] += res.query_time
                totals["total_time"] += res.total_time
                totals["insert_comps"] += getattr(res, 'insert_comparisons', 0)
                totals["query_comps"] += getattr(res, 'query_comparisons', 0)
                totals["total_comps"] += getattr(res, 'insert_comparisons', 0) + getattr(res, 'query_comparisons', 0)
                totals["p99"] += getattr(res, 'p99_query_time', 0.0)
                totals["memory"] = max(totals["memory"], res.total_memory_kb / 1024.0)

                if mode == "interleaved_delete":
                    total_ops = res.p99_query_time
                    totals["throughput"] += (total_ops / res.total_time) if res.total_time > 0 else 0
                    totals["p99"] += 0.0
                else:
                    totals["p99"] += getattr(res, 'p99_query_time', 0.0)
                    if mode == "interleaved":
                        totals["throughput"] += (
                                    (insert_count + len(test_queries)) / res.total_time) if res.total_time > 0 else 0
                    else:
                        totals["qps"] += (len(test_queries) / res.query_time) if res.query_time > 0 else 00

            avg = {k: (v / runs if k != "memory" else v) for k, v in totals.items()}
            mem_str = f"{avg['memory']:.1f}"

            if mode in ["interleaved", "interleaved_delete"]:
                results_avg.append((
                    tree, avg['total_time'], int(avg['throughput']), avg['p99'],
                    int(avg['insert_comps']), int(avg['query_comps']), int(avg['total_comps']),
                    mem_str, status
                ))
                perf_metric = int(avg['throughput'])
            else:
                results_avg.append((
                    tree, avg['insert_time'], avg['query_time'], avg['total_time'],
                    int(avg['insert_comps']), int(avg['query_comps']), int(avg['total_comps']),
                    int(avg['qps']), mem_str, status
                ))
                perf_metric = int(avg['qps'])

            writer.writerow([
                CURRENT_BENCH_TYPE, tree, runs,
                f"{avg['insert_time']:.6f}", f"{avg['query_time']:.6f}", f"{avg['total_time']:.6f}",
                int(avg['insert_comps']), int(avg['query_comps']), int(avg['total_comps']),
                f"{avg['p99']:.6f}", perf_metric, mem_str, status
            ])

            gc.collect()

    print("[SUCCESS] All tree variants evaluated and averaged.                 ")

    if mode in ["interleaved", "interleaved_delete"]:
        print_interleaved_table(results_avg)
    else:
        print_batched_table(results_avg)


def run_standard(args):
    total_n = args.preload + args.insert
    print(f"\n[INFO] Generating Standard Synthetic Data ({total_n:,} keys)...")
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    train_queries = generate_standard_queries(args.q, total_n, args.dist, args.zipf_alpha, all_keys)
    train_and_export_oracle(np.array(train_queries))

    test_queries = generate_standard_queries(args.q, total_n, args.dist, args.zipf_alpha, all_keys)

    params = {
        "Type": "Standard Synthetic (Unified)",
        "Preload": f"{len(preload_list):,}",
        "Inserts": f"{len(insert_list):,}",
        "Queries": f"{len(test_queries):,}",
        "Distribution": f"Zipfian (alpha={args.zipf_alpha})" if args.dist == 'zipf' else "Uniform"
    }
    print_header("UNIFIED SYNTHETIC BENCHMARK", params)
    evaluate_trees(preload_list, insert_list, test_queries, mode="batched")


def run_mixture(args):
    total_n = args.preload + args.insert
    print(
        f"\n[INFO] Generating Mixture Data ({args.mix_ratio * 100}% Zipfian, {100 - args.mix_ratio * 100}% Uniform)...")

    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    train_queries = generate_mixed_queries(args.q, total_n, args.zipf_alpha, args.mix_ratio, all_keys)
    train_and_export_oracle(np.array(train_queries))

    test_queries = generate_mixed_queries(args.q, total_n, args.zipf_alpha, args.mix_ratio, all_keys)

    params = {
        "Type": f"Mixture Distribution ({args.mix_ratio * 100}% Zipf)",
        "Preload": f"{len(preload_list):,}",
        "Inserts": f"{len(insert_list):,}",
        "Queries": f"{len(test_queries):,}",
        "Zipf Alpha": args.zipf_alpha
    }
    print_header("MIXTURE BENCHMARK", params)

    evaluate_trees(preload_list, insert_list, test_queries, mode="batched")

    
def run_skew_drift(args):
    total_n = args.preload + args.insert
    print(f"\n[INFO] Generating Data for Skew Drift Benchmark ({total_n:,} keys)...")
    
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    # Train the Oracle on a mild Distribution
    train_alpha = 1.01  # Almost uniform, minimal skew
    print(f"\n[INFO] --- PHASE 1: Training Oracle on Mild Zipfian (alpha={train_alpha}) ---")
    train_queries = generate_standard_queries(args.q, total_n, 'zipf', train_alpha, all_keys)
    
    train_and_export_oracle(np.array(train_queries))

    # Test against severe skew
    test_alphas = [1.01, 1.25, 1.5, 2.0, 3.0]
    print(f"\n[INFO] --- PHASE 2: Testing Concept Drift against alphas: {test_alphas} ---")

    for test_alpha in test_alphas:
        # Generate the new, steeper workload
        test_queries = generate_standard_queries(args.q, total_n, 'zipf', test_alpha, all_keys)

        params = {
            "Type": f"Skew Drift (Train={train_alpha}, Test={test_alpha})",
            "Preload": f"{len(preload_list):,}",
            "Inserts": f"{len(insert_list):,}",
            "Queries": f"{len(test_queries):,}",
            "Train Skew": f"alpha={train_alpha}",
            "Test Skew": f"alpha={test_alpha}"
        }
        
        print_header(f"SKEW DRIFT BENCHMARK (Test Alpha: {test_alpha})", params)
        
        # Evaluate without retraining
        evaluate_trees(preload_list, insert_list, test_queries, mode="batched")


def run_interleaved(args):
    total_n = args.preload + args.insert
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    train_queries = generate_standard_queries(args.q, total_n, args.dist, args.zipf_alpha, all_keys)
    train_and_export_oracle(np.array(train_queries))

    test_queries = generate_standard_queries(args.q, total_n, args.dist, args.zipf_alpha, all_keys)

    params = {
        "Type": "Interleaved (Unified)",
        "Preload": f"{args.preload:,}",
        "Inserts": f"{args.insert:,}",
        "Queries (Approx)": f"{args.q:,}"
    }
    print_header("UNIFIED INTERLEAVED", params)
    evaluate_trees(preload_list, insert_list, test_queries, mode="interleaved", insert_count=args.insert)


def run_interleaved_delete(args):
    total_n = args.preload + args.insert
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    train_queries = generate_standard_queries(args.q, total_n, args.dist, args.zipf_alpha, all_keys)
    train_and_export_oracle(np.array(train_queries))

    test_queries = generate_standard_queries(args.q, total_n, args.dist, args.zipf_alpha, all_keys)

    params = {
        "Type": "Interleaved Deletes (Unified)",
        "Preload": f"{args.preload:,}",
        "Insert Pool": f"{args.insert:,}",
        "Target Queries": f"{args.q:,}"
    }
    print_header("UNIFIED INTERLEAVED DELETES", params)
    evaluate_trees(preload_list, insert_list, test_queries, mode="interleaved_delete", insert_count=args.insert)


def run_blackhole(args):
    N_total = args.preload + args.insert
    all_keys = random.sample(range(N_total * 10), N_total)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    valid_keys = [k for k in all_keys if k <= args.range]
    if not valid_keys:
        valid_keys = all_keys[:100]

    train_queries = [random.choice(valid_keys) if random.random() < 0.50 else random.choice(all_keys) for _ in
                     range(args.q)]
    train_and_export_oracle(np.array(train_queries))

    test_queries = [random.choice(valid_keys) for _ in range(args.q)]

    params = {
        "Type": "Black Hole (Unified)",
        "Preload": f"{len(preload_list):,}",
        "Inserts": f"{len(insert_list):,}",
        "Queries": f"{len(test_queries):,}",
        "Query Range": f"[0, {args.range}]"
    }
    print_header("BLACK HOLE BENCHMARK", params)
    evaluate_trees(preload_list, insert_list, test_queries, mode="batched")


def run_shifting_hotspot(args):
    total_n = args.preload + args.insert
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    phase_size, hot_a = args.q // 3, int(total_n * 0.1)
    train_queries = [random.randint(hot_a, hot_a + 1000) if random.random() < 0.8 else random.randint(0, total_n - 1)
                     for _ in range(args.q)]
    train_and_export_oracle(np.array(train_queries))

    test_queries = []
    test_queries.extend(
        [random.randint(hot_a, hot_a + 1000) if random.random() < 0.8 else random.randint(0, total_n - 1) for _ in
         range(phase_size)])
    hot_b = int(total_n * 0.9)
    test_queries.extend(
        [random.randint(hot_b, hot_b + 1000) if random.random() < 0.8 else random.randint(0, total_n - 1) for _ in
         range(phase_size)])
    hot_c = int(total_n * 0.5)
    test_queries.extend(
        [random.randint(hot_c, hot_c + 1000) if random.random() < 0.8 else random.randint(0, total_n - 1) for _ in
         range(args.q - 2 * phase_size)])

    params = {
        "Type": "Shifting Hotspot (Unified)",
        "Preload": f"{len(preload_list):,}",
        "Inserts": f"{len(insert_list):,}",
        "Queries": f"{len(test_queries):,}"
    }
    print_header("SHIFTING HOTSPOT BENCHMARK", params)
    evaluate_trees(preload_list, insert_list, test_queries, mode="batched")


def run_corrupted(args):
    total_n = args.preload + args.insert
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    train_queries = generate_standard_queries(args.q, total_n, 'zipf', args.zipf_alpha, all_keys)
    train_and_export_oracle(np.array(train_queries))

    corrupted_keys = list(reversed(all_keys))
    test_queries = generate_standard_queries(args.q, total_n, 'zipf', args.zipf_alpha, corrupted_keys)

    params = {
        "Type": "Corrupted Workload (Unified)",
        "Distribution": f"Zipfian (alpha={args.zipf_alpha}) - INVERTED!"
    }
    print_header("CORRUPTED WORKLOAD BENCHMARK", params)
    evaluate_trees(preload_list, insert_list, test_queries, mode="batched")


def run_crossover(args):
    total_n = args.preload + args.insert
    all_keys = list(range(total_n))
    random.shuffle(all_keys)
    preload_list, insert_list = all_keys[:args.preload], all_keys[args.preload:]

    base_qs = [100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    schedule = [q for q in base_qs if q <= args.q]
    if args.q not in schedule:
        schedule.append(args.q)

    master_train_queries = generate_standard_queries(schedule[-1], total_n, args.dist, args.zipf_alpha, all_keys)
    train_and_export_oracle(np.array(master_train_queries))

    master_test_queries = generate_standard_queries(schedule[-1], total_n, args.dist, args.zipf_alpha, all_keys)

    params = {
        "Type": "Crossover Threshold",
        "Preload": f"{len(preload_list):,}",
        "Inserts": f"{len(insert_list):,}"
    }
    print_header("CROSSOVER BENCHMARK", params)

    target_dir = os.path.join(ROOT_DIR, "results_synthetic")
    os.makedirs(target_dir, exist_ok=True)
    csv_file = os.path.join(target_dir, f"crossover_results_{args.dist}_{args.q}q.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Queries (Q)", "OrigLazy Time (s)", "LearnLazy Time (s)", "Winner"])

        row_print = True
        for q in schedule:
            test_queries = master_test_queries[:q]

            lazy_orig = run_tree_benchmark("OriginalLazy", preload_list, insert_list, test_queries)
            lazy_learn = run_tree_benchmark("LearnedLazy", preload_list, insert_list, test_queries)

            times = {"OrigLazy": lazy_orig.total_time, "LearnLazy": lazy_learn.total_time}
            winner = min(times.keys(), key=lambda k: times[k])

            if row_print:
                print(f"\n| {'Queries (Q)':<12} | {'OrigLazy Time':<15} | {'LearnLazy Time':<15} | {'Winner':<10} |")
                print("-" * 63)
                row_print = False

            print(f"| {q:<12,} | {times['OrigLazy']:<15.4f} | {times['LearnLazy']:<15.4f} | {winner:<10} |")

            writer.writerow([q, f"{times['OrigLazy']:.6f}", f"{times['LearnLazy']:.6f}", winner])

    print(f"\n[SUCCESS] Crossover results exported to: {csv_file}\n")


def main():
    global GLOBAL_ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench',
                        choices=['synthetic', 'blackhole', 'interleaved', 'interleaved_delete', 
                                 'shifting', 'crossover', 'corrupted', 'skew_drift', 'mixture'],
                        default='synthetic')
    parser.add_argument('--preload', type=int, default=1000000)
    parser.add_argument('--insert', type=int, default=100000)
    parser.add_argument('--q', type=int, default=100000)
    parser.add_argument('--dist', choices=['uniform', 'zipf'], default='zipf')
    parser.add_argument('--zipf_alpha', type=float, default=1.01)
    parser.add_argument('--range', type=int, default=5000)
    parser.add_argument('--runs', type=int, default=1, help="Number of times to run and average")
    parser.add_argument('--mix_ratio', type=float, default=0.25, help="Ratio of Zipfian queries in the mixture")
    args = parser.parse_args()

    GLOBAL_ARGS = args

    if args.bench == 'synthetic':
        run_standard(args)
    elif args.bench == 'skew_drift':
        run_skew_drift(args)
    elif args.bench == 'mixture':
        run_mixture(args)
    elif args.bench == 'blackhole':
        run_blackhole(args)
    elif args.bench == 'interleaved':
        run_interleaved(args)
    elif args.bench == 'interleaved_delete':
        run_interleaved_delete(args)
    elif args.bench == 'shifting':
        run_shifting_hotspot(args)
    elif args.bench == 'crossover':
        run_crossover(args)
    elif args.bench == 'corrupted':
        run_corrupted(args)


if __name__ == "__main__":
    main()
