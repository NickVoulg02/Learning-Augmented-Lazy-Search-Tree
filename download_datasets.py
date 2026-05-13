import kagglehub
import pandas as pd
import os
import glob
import argparse
import re
from collections import Counter


def download_bbc(top_k=5500):
    print("\n" + "=" * 52)
    print("[INFO] Starting BBC News Dataset Download & Setup")
    print("=" * 52)

    try:
        path = kagglehub.dataset_download("shivamkushwaha/bbc-full-text-document-classification")
        print(f"[SUCCESS] Dataset downloaded to: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        return None

    all_files = glob.glob(os.path.join(path, "**/*.txt"), recursive=True)
    if not all_files:
        print("[ERROR] No text files found.")
        return None

    print(f"[INFO] Found {len(all_files)} text files. Tokenizing...")

    words_sequence = []

    for file in all_files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().lower()
                tokens = re.findall(r'\b[a-z]+\b', text)
                words_sequence.extend(tokens)
        except Exception as e:
            print(f"[WARN] Failed to read {file}: {e}")

    word_counts = Counter(words_sequence)
    top_words = [word for word, count in word_counts.most_common(top_k)]

    print(f"[INFO] Total words processed: {len(words_sequence):,}")
    print(f"[INFO] Extracted top {top_k} vocabulary words.")

    word_to_key = {word: i + 1 for i, word in enumerate(top_words)}

    query_log = []
    for word in words_sequence:
        if word in word_to_key:
            query_log.append(word_to_key[word])

    df_out = pd.DataFrame({"Key": query_log})

    output_file = "bbc_dataset.csv"
    df_out.to_csv(output_file, index=False)

    print("-" * 52)
    print(f"[SUCCESS] Saved {len(df_out):,} integer queries to '{output_file}'")
    print("-" * 52 + "\n")

    return output_file


def download_network():
    print("\n" + "=" * 52)
    print("[INFO] Starting CIC-IDS2017 Dataset Download & Setup")
    print("=" * 52)

    try:
        path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
        print(f"[SUCCESS] Dataset downloaded to: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        return None

    all_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    if not all_files:
        print("[ERROR] No CSV files found in the dataset.")
        return None

    print(f"[INFO] Found {len(all_files)} CSV files.")

    target_columns = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets"]
    print(f"\n[INFO] Merging relevant traffic data for columns:")
    print(f"       -> {', '.join(target_columns)}")

    merged_data = []
    for file in all_files:
        filename = os.path.basename(file)
        print(f"[PROCESS] Reading {filename}...", end='\r')
        try:
            chunk_iter = pd.read_csv(file, chunksize=100000, encoding='utf-8', on_bad_lines='skip')
            for chunk in chunk_iter:
                chunk.columns = chunk.columns.str.strip()
                if all(col in chunk.columns for col in target_columns):
                    filtered_chunk = chunk[target_columns].dropna()
                    merged_data.append(filtered_chunk)
                else:
                    break
        except Exception as e:
            print(f"\n[WARN] Skipping {filename} due to error: {e}")

    print("\n[SUCCESS] All files processed.                 ")

    if not merged_data:
        print("[ERROR] Could not extract targeted data.")
        return None

    df_out = pd.concat(merged_data, ignore_index=True)
    df_out["Key"] = df_out["Destination Port"].astype(int)

    df_out = df_out.drop(columns=["Destination Port"])

    output_file = "network_traffic.csv"
    print(f"[INFO] Total Records Extracted: {len(df_out):,}")
    print(f"[INFO] Saving merged data to '{output_file}'...")
    df_out.to_csv(output_file, index=False)

    print("[SUCCESS] Data preparation complete!\n")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Dataset Downloader")
    parser.add_argument('--dataset', choices=['network', 'bbc'], default='network',
                        help="Choose which dataset to download and process.")
    parser.add_argument('--vocab_size', type=int, default=5500,
                        help="Number of top words to map to integer keys (BBC only)")
    args = parser.parse_args()

    if args.dataset == 'network':
        download_network()
    elif args.dataset == 'bbc':
        download_bbc(top_k=args.vocab_size)
