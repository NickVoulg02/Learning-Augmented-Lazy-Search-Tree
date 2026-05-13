import subprocess

param_sets = [
    ("1000000", "100000", "2500"),
    ("1000000", "100000", "25000"),
    ("1000000", "200000", "2500"),
    ("1000000", "1000000", "25000"),
    ("1000000", "1000000", "250000"),
    ("1000000", "2000000", "25000")
]

target_script = "shared/benchmark_synthetic_unified.py"
benchmarks = ["blackhole", "ood", "skew_drift", "shifting", "interleaved", "interleaved_delete", "corrupted", "crossover"]


zipf_alphas = ["1.5"]
RUNS = "10"

for preload, insert, query in param_sets:
    print("\n" + "=" * 70)
    print(f" RUNNING SUITE -> Preload: {preload} | Inserts: {insert} | Queries: {query}")
    print("=" * 70)

    cmd_uniform = [
        "python", target_script, "--bench", "synthetic", "--dist", "uniform",
        "--preload", preload, "--insert", insert, "--q", query, "--runs", RUNS
    ]
    print(f"\n>>> Executing: {' '.join(cmd_uniform)}")
    subprocess.run(cmd_uniform)

    for alpha in zipf_alphas:
        cmd_zipf = [
            "python", target_script, "--bench", "synthetic", "--dist", "zipf",
            "--zipf_alpha", alpha, "--preload", preload, "--insert", insert, "--q", query, "--runs", RUNS
        ]
        print(f"\n>>> Executing: {' '.join(cmd_zipf)}")
        subprocess.run(cmd_zipf)

    cmd_mix = [
        "python", target_script, "--bench", "mixture", "--mix_ratio", "0.25",
        "--zipf_alpha", "1.5", "--preload", preload, "--insert", insert, "--q", query, "--runs", RUNS
    ]
    print(f"\n>>> Executing: {' '.join(cmd_mix)}")
    subprocess.run(cmd_mix)

    for bench in benchmarks:
        cmd_adv = [
            "python", target_script, "--bench", bench,
            "--preload", preload, "--insert", insert, "--q", query, "--runs", RUNS
        ]
        print(f"\n>>> Executing: {' '.join(cmd_adv)}")
        subprocess.run(cmd_adv)

print("\n" + "=" * 70)
print("[SUCCESS] All synthetic benchmark suites completed.")
print("=" * 70)
