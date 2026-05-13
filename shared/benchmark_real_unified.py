import gc
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(ROOT_DIR)

import argparse
import csv
import random
from collections import Counter

import lazy_tree_original as lst_orig
import lazy_tree_learned as lst_learn

TREE_VARIANTS = ["BTree", "Splay", "Treap", "OriginalLazy", "LearnedLazy"]

GLOBAL_ARGS = None
CURRENT_BENCH_TYPE = "Unknown"


def check_oracle_weights():
    weight_path = os.path.abspath(os.path.join(ROOT_DIR, "src", "learned", "weights.txt"))
    if not os.path.exists(weight_path):
        print(f"\n[CRITICAL ERROR] '{weight_path}' not found!")
        print("Please run 'python src/learned/cdf_oracle.py' first to train the model and export weights.")
        print("The Learned Lazy Tree cannot function without the Neural Network weights.\n")
        exit(1)


def run_tree_benchmark(tree_name, preload, inserts, queries, is_interleaved=False):
    if tree_name in ["BTree", "Splay", "OriginalLazy"]:
        backend = lst_orig
    else:
        backend = lst_learn

    internal_name = "Lazy" if "Lazy" in tree_name else tree_name

    if is_interleaved:
        return backend.run_custom_interleaved(internal_name, preload, inserts, queries)
    else:
        return backend.run_custom(internal_name, preload, inserts, queries)


def get_real_data(limit, dataset_type):
    filename = "bbc_dataset.csv" if dataset_type == 'bbc' else "network_traffic.csv"
    full_path = os.path.join(ROOT_DIR, filename)
    data = []
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    key_idx = next((i for i, h in enumerate(headers) if "Key" in h.strip()), -1)
                    if key_idx != -1:
                        for row in reader:
                            if len(row) > key_idx:
                                try:
                                    data.append(int(row[key_idx]))
                                except ValueError:
                                    continue
                            if len(data) >= limit:
                                break
        except Exception as e:
            print(f"[ERROR] Failed reading {filename}: {e}")

    if not data:
        print(f"[WARN] File not found. Generating fallback synthetic {dataset_type} data.")
        for _ in range(limit):
            if dataset_type == 'network':
                data.append(
                    random.choice([80, 443, 8080, 22, 53]) if random.random() < 0.8 else random.randint(0, 65535))
            else:
                data.append(random.randint(0, 12620200))
    return data


def get_query_data(limit, filename="query_workload_test.csv"):
    queries = []
    full_path = os.path.join(ROOT_DIR, filename)
    if os.path.exists(full_path):
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers and "Key" in headers[0]:
                    for row in reader:
                        if row:
                            try:
                                queries.append(int(row[0]))
                            except ValueError:
                                continue
                            if len(queries) >= limit:
                                break
        except Exception as e:
            print(f"[ERROR] Failed reading {filename}: {e}")

    if not queries:
        print(f"[WARN] '{filename}' not found. Using random fallback queries.")
        queries = [random.randint(0, 65535) for _ in range(limit)]

    if len(queries) < limit:
        repeat_factor = (limit // len(queries)) + 1
        queries = (queries * repeat_factor)[:limit]

    return queries


def print_distribution_compact(data, top_n=5):
    if not data:
        return
    counts = Counter(data)
    top_str = ", ".join([f"{p} ({c / len(data):.1%})" for p, c in counts.most_common(top_n)])
    print(f"[INFO] Base Dataset: {len(data):,} items, {len(counts):,} unique keys.")
    print(f"       Top {top_n}: {top_str}")


def print_header(title, params):
    global CURRENT_BENCH_TYPE
    CURRENT_BENCH_TYPE = params.get("Type", title)

    print("\n" + "=" * 60)
    print(f"[BENCHMARK] {title}")
    print("=" * 60)
    for k, v in params.items():
        print(f"  -> {k:<18}: {v}")
    print("-" * 60)


def evaluate_trees(preload_list, insert_list, test_queries, is_interleaved=False, insert_count=0):
    global GLOBAL_ARGS, CURRENT_BENCH_TYPE

    runs = getattr(GLOBAL_ARGS, 'runs', 1)
    dataset = getattr(GLOBAL_ARGS, 'dataset', getattr(GLOBAL_ARGS, 'bench', 'synthetic'))
    ins = getattr(GLOBAL_ARGS, 'insert', 0)
    q = getattr(GLOBAL_ARGS, 'q', 0)

    target_dir = os.path.join(ROOT_DIR, "results_real")
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
                res = run_tree_benchmark(tree, preload_list, insert_list, test_queries, is_interleaved)

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

                if is_interleaved:
                    totals["throughput"] += ((insert_count + len(test_queries)) / res.total_time) if res.total_time > 0 else 0
                else:
                    totals["qps"] += (len(test_queries) / res.query_time) if res.query_time > 0 else 0

            avg = {k: (v / runs if k != "memory" else v) for k, v in totals.items()}
            mem_str = f"{avg['memory']:.1f}"

            if is_interleaved:
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

    if is_interleaved:
        print("\n" + "=" * 140)
        print(
            f"| {'Tree Variant':<14} | {'Total Time(s)':<13} | {'Throughput':<11} | {'Ins Comps':<13} | {'Qry Comps':<13} | {'Tot Comps':<13} | {'Mem(MB)':<10} |")
        print("-" * 140)
        for r in results_avg:
            print(
                f"| {r[0]:<14} | {r[1]:<13.4f} | {r[2]:<11,} | {r[4]:<13,} | {r[5]:<13,} | {r[6]:<13,} | {r[7]:<10} |")
        print("=" * 140 + "\n")
    else:
        print("\n" + "=" * 145)
        print(
            f"| {'Tree Variant':<14} | {'Ins Time(s)':<11} | {'Qry Time(s)':<11} | {'Total (s)':<12} | {'Ins Comps':<13} | {'Qry Comps':<13} | {'Tot Comps':<13} | {'Mem(MB)':<10} |")
        print("-" * 145)
        for r in results_avg:
            print(
                f"| {r[0]:<14} | {r[1]:<11.4f} | {r[2]:<11.4f} | {r[3]:<12.4f} | {r[4]:<13,} | {r[5]:<13,} | {r[6]:<13,} | {r[8]:<10} |")
        print("=" * 145 + "\n")


def run_standard_real(args):
    check_oracle_weights()
    mode_str = "INTERLEAVED" if args.interleaved else "BATCHED"

    total_base_needed = args.preload + args.insert
    print(f"\n[INFO] Loading {total_base_needed:,} base records from {args.dataset}...")
    base_data = get_real_data(total_base_needed, args.dataset)

    if len(base_data) < total_base_needed:
        base_data = (base_data * ((total_base_needed // len(base_data)) + 1))[:total_base_needed]

    preload_data = base_data[:args.preload]
    insert_data = base_data[args.preload: args.preload + args.insert]

    print_distribution_compact(preload_data)
    print(f"\n[INFO] Loading {args.q:,} queries from workload log...")
    query_data = get_query_data(args.q, "query_workload_test.csv")

    params = {
        "Type": f"Standard Real ({mode_str})",
        "Dataset": args.dataset.capitalize(),
        "Preload": f"{args.preload:,}", "Inserts": f"{args.insert:,}", "Queries": f"{args.q:,}"
    }
    print_header(f"UNIFIED REAL DATA", params)

    evaluate_trees(preload_data, insert_data, query_data, args.interleaved, args.insert)


def run_mirage(args):
    check_oracle_weights()
    preload_data = get_real_data(args.preload, args.dataset)
    if len(preload_data) < args.preload:
        preload_data = (preload_data * ((args.preload // len(preload_data)) + 1))[:args.preload]

    real_max = max(preload_data)
    top_10_targets = [item[0] for item in Counter(get_query_data(500000, "query_workload_test.csv")).most_common(10)]
    dense_target = top_10_targets[0]

    chasm_offset, chasm_size = (10000, 5000) if args.dataset == 'bbc' else (20000, 10000)
    chasm_min, chasm_max = real_max + chasm_offset, real_max + chasm_offset + chasm_size

    insert_data = [random.randint(chasm_min, chasm_max) for _ in range(10000)] + [dense_target] * 10
    random.shuffle(insert_data)
    query_data = [random.choice(top_10_targets) for _ in range(args.q)]

    params = {
        "Type": "Adversarial Mirage",
        "Dataset": args.dataset.capitalize(),
        "Dense Target": f"{dense_target:,}",
        "Mirage Range": f"[{chasm_min:,}, {chasm_max:,}]"
    }
    print_header(f"MIRAGE TRAP BENCHMARK", params)
    evaluate_trees(preload_data, insert_data, query_data)


def run_crossover_real(args):
    check_oracle_weights()

    total_base_needed = args.preload + args.insert
    print(f"\n[INFO] Loading {total_base_needed:,} base records from {args.dataset}...")
    base_data = get_real_data(total_base_needed, args.dataset)

    if len(base_data) < total_base_needed:
        base_data = (base_data * ((total_base_needed // len(base_data)) + 1))[:total_base_needed]

    preload_data = base_data[:args.preload]
    insert_data = base_data[args.preload: args.preload + args.insert]

    base_qs = [100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]
    schedule = [q for q in base_qs if q <= args.q]
    if args.q not in schedule:
        schedule.append(args.q)

    print(f"\n[INFO] Loading {schedule[-1]:,} queries from workload log...")
    master_test_queries = get_query_data(schedule[-1], "query_workload_test.csv")

    params = {
        "Type": "Crossover Threshold (Real Data)",
        "Dataset": args.dataset.capitalize(),
        "Preload": f"{len(preload_data):,}",
        "Inserts": f"{len(insert_data):,}"
    }
    print_header("CROSSOVER BENCHMARK", params)

    target_dir = os.path.join(ROOT_DIR, "results_real")
    os.makedirs(target_dir, exist_ok=True)
    csv_file = os.path.join(target_dir, f"crossover_results_{args.dataset}_{args.q}q.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Queries (Q)", "OrigLazy Time (s)", "LearnLazy Time (s)", "Winner"])

        row_print = True
        for q in schedule:
            test_queries = master_test_queries[:q]

            lazy_orig = run_tree_benchmark("OriginalLazy", preload_data, insert_data, test_queries)
            lazy_learn = run_tree_benchmark("LearnedLazy", preload_data, insert_data, test_queries)

            times = {"OrigLazy": lazy_orig.total_time, "LearnLazy": lazy_learn.total_time}
            winner = min(times.keys(), key=lambda k: times[k])

            if row_print:
                print(f"\n| {'Queries (Q)':<12} | {'OrigLazy Time':<15} | {'LearnLazy Time':<15} | {'Winner':<10} |")
                print("-" * 63)
                row_print = False

            print(f"| {q:<12,} | {times['OrigLazy']:<15.4f} | {times['LearnLazy']:<15.4f} | {winner:<10} |")

            writer.writerow([q, f"{times['OrigLazy']:.6f}", f"{times['LearnLazy']:.6f}", winner])

    print(f"\n[SUCCESS] Crossover results exported to: {csv_file}\n")


def run_narrow_range(args):
    check_oracle_weights()

    total_base_needed = args.preload + args.insert
    print(f"\n[INFO] Loading {total_base_needed:,} base records from {args.dataset}...")
    base_data = get_real_data(total_base_needed, args.dataset)

    if len(base_data) < total_base_needed:
        base_data = (base_data * ((total_base_needed // len(base_data)) + 1))[:total_base_needed]

    preload_data = base_data[:args.preload]

    print(f"\n[INFO] Analyzing query log to find the absolute Hotspot...")
    full_query_log = get_query_data(500000, "query_workload_test.csv")

    top_target = Counter(full_query_log).most_common(1)[0][0]

    variance = 100
    min_target = max(0, top_target - variance)
    max_target = top_target + variance

    print(f"[INFO] Hotspot found: {top_target}. Creating a narrow range [{min_target}, {max_target}]...")
    print(f"[INFO] Redirecting 100% of Inserts and Queries to this small range...")

    insert_data = [random.randint(min_target, max_target) for _ in range(args.insert)]
    query_data = [random.randint(min_target, max_target) for _ in range(args.q)]

    params = {
        "Type": "Narrow Range",
        "Dataset": args.dataset.capitalize(),
        "Target Range": f"[{min_target:,}, {max_target:,}]",
        "Preload": f"{args.preload:,}",
        "Inserts": f"{args.insert:,} (100% to Range)",
        "Queries": f"{args.q:,} (100% to Range)"
    }
    print_header(f"NARROW RANGE BENCHMARK", params)

    evaluate_trees(preload_data, insert_data, query_data, is_interleaved=args.interleaved, insert_count=args.insert)


def run_corrupted_real(args):
    check_oracle_weights()
    total_base_needed = args.preload + args.insert
    base_data = get_real_data(total_base_needed, args.dataset)
    if len(base_data) < total_base_needed:
        base_data = (base_data * ((total_base_needed // len(base_data)) + 1))[:total_base_needed]

    preload_data, insert_data = base_data[:args.preload], base_data[args.preload: args.preload + args.insert]

    standard_query_data = get_query_data(args.q, "query_workload_test.csv")

    print(f"\n[INFO] Corrupting workload (Swapping Hot/Cold Ranks)...")
    sorted_keys = [k for k, v in Counter(standard_query_data).most_common()]
    reversed_keys = list(reversed(sorted_keys))
    corruption_map = dict(zip(sorted_keys, reversed_keys))

    corrupted_query_data = [corruption_map[k] for k in standard_query_data]

    params = {
        "Type": "Adversarial Rank Swapping",
        "Dataset": args.dataset.capitalize(),
        "Queries": f"{args.q:,}",
        "Note": f"Hottest key '{sorted_keys[0]}' swapped with '{reversed_keys[0]}'" if sorted_keys else "N/A"
    }
    print_header(f"CORRUPTED ORACLE BENCHMARK", params)
    evaluate_trees(preload_data, insert_data, corrupted_query_data)


def main():
    global GLOBAL_ARGS
    parser = argparse.ArgumentParser(description="Unified Real Data Benchmark")
    parser.add_argument('--dataset', choices=['bbc', 'network'], default='network', help="Choose the dataset.")
    parser.add_argument('--preload', type=int, default=1000000, help="Initial tree size")
    parser.add_argument('--insert', type=int, default=100000, help="New items to insert")
    parser.add_argument('--q', type=int, default=50000, help="Number of queries")
    parser.add_argument('--interleaved', action='store_true', help="Run interleaved benchmark")
    parser.add_argument('--crossover', action='store_true', help="Run Crossover Threshold benchmark")
    parser.add_argument('--mirage', action='store_true', help="Run Mirage trap")
    parser.add_argument('--corrupted', action='store_true', help="Run Adversarial Rank Swapping test")
    parser.add_argument('--pq', action='store_true', help="Run Priority Queue simulation with Narrow Range")
    parser.add_argument('--runs', type=int, default=1, help="Number of times to run and average")
    args = parser.parse_args()

    GLOBAL_ARGS = args

    if args.pq:
        run_narrow_range(args)
    elif args.mirage:
        run_mirage(args)
    elif args.crossover:
        run_crossover_real(args)
    elif args.corrupted:
        run_corrupted_real(args)
    else:
        run_standard_real(args)


if __name__ == "__main__":
    main()
