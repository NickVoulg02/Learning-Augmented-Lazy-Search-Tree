import subprocess

datasets = ['network']
RUNS = "10"
target_script = "shared/benchmark_real_unified.py"

param_sets = [
    ("10000", "100000", "2500"),
    ("10000", "100000", "25000"),
    ("10000", "200000", "2500"),
    ("10000", "1000000", "25000"),
    ("10000", "1000000", "250000"),
    ("10000", "2000000", "25000")
]

for dataset in datasets:
    print("\n" + "#" * 70)
    print(f" EVALUATING DATASET: {dataset.upper()}")
    print("#" * 70)

    for preload, insert, query in param_sets:
        print("\n" + "=" * 60)
        print(f"[*] Scale -> Preload: {preload} | Inserts: {insert} | Queries: {query}")
        print("=" * 60)

        base_cmd = [
            "python", target_script, "--dataset", dataset,
            "--preload", preload, "--insert", insert, "--q", query, "--runs", RUNS
        ]

        print("\n>>> Running Standard (Batched)...")
        subprocess.run(base_cmd)

        print("\n>>> Running Narrow Range PQ (Batched)...")
        subprocess.run(base_cmd + ["--pq"])

        print(f"\n[*] Running Interleaved ({dataset})...")
        subprocess.run(base_cmd + ["--interleaved"])

        print(f"\n[*] Running Hotspot ({dataset})...")
        subprocess.run(base_cmd + ["--hotspot"])

        print("\n>>> Running Mirage Adversarial...")
        subprocess.run(base_cmd + ["--mirage"])

        print(f"\n[*] Running Corrupted ({dataset})...")
        subprocess.run(base_cmd + ["--corrupted"])

print("\n" + "=" * 70)
print("[SUCCESS] All real-data benchmark suites completed.")
print("=" * 70)
