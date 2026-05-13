import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import argparse

from cdf_dataset import CDFDataset
from cdf_model import CDFNet


def calculate_max_error(model, loader, device):
    model.eval()
    max_err = 0.0
    total_mae = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets_cdf in loader:
            inputs, targets_cdf = inputs.to(device), targets_cdf.to(device)

            preds_cdf = model(inputs)

            errors = torch.abs(preds_cdf - targets_cdf)
            batch_max = errors.max().item()
            if batch_max > max_err:
                max_err = batch_max

            total_mae += errors.sum().item()
            count += len(targets_cdf)

    return max_err, total_mae / count


def export_weights_to_cpp(model, max_val, filename="weights.txt"):
    print(f"\n[EXPORT] Exporting weights to '{filename}'")
    with open(filename, 'w') as f:
        f.write(f"MAX_KEY {max_val}\n")

        for name, param in model.named_parameters():
            if 'weight' in name:
                f.write(f"WEIGHT {name}\n")
                weights = param.detach().cpu().numpy()

                for row in weights:
                    f.write(" ".join([f"{x:.8f}" for x in row]) + "\n")

            elif 'bias' in name:
                f.write(f"BIAS {name}\n")
                data = param.detach().cpu().numpy().flatten()

                f.write(" ".join([f"{x:.8f}" for x in data]) + "\n")


def train(args):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    dataset_path = os.path.join(repo_root, args.dataset) if not os.path.isabs(args.dataset) else args.dataset
    weights_path = os.path.join(current_dir, "weights.txt")
    # -----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 55)
    print(f"[TRAIN] Initializing Training on Device: {str(device).upper()}")
    print("=" * 55)
    print(f"  -> Dataset:    {dataset_path}")
    print(f"  -> Output:     {weights_path}")
    print(f"  -> Epochs:     {args.epochs}")
    print(f"  -> Batch Size: {args.batch}")
    print(f"  -> Learn Rate: {args.lr}")
    print(f"  -> Hidden Dim: {args.hidden_dim}")
    print("-" * 55)

    full_dataset = CDFDataset(filepath=dataset_path, num_items=500000)

    train_loader = DataLoader(full_dataset, batch_size=args.batch, shuffle=True)

    model = CDFNet(hidden_dim=args.hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    start_time = time.time()
    print("[TRAIN] Starting Epochs...")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets_cdf in train_loader:
            inputs = inputs.to(device)
            targets_cdf = targets_cdf.to(device)

            optimizer.zero_grad()

            preds_cdf = model(inputs)
            loss = criterion(preds_cdf, targets_cdf)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Evaluate error on the training set to guide the LR scheduler
        max_err, avg_mae = calculate_max_error(model, train_loader, device)
        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"        -> Epoch {epoch + 1:02d}/{args.epochs} | Loss: {avg_loss:.6f} | MaxErr: {max_err:.4f} | AvgErr: {avg_mae:.4f} | LR: {current_lr:.1e}")

    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Training finished in {elapsed_time:.2f}s")

    export_weights_to_cpp(model, full_dataset.max_key, filename=weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Learned Query CDF Oracle")
    parser.add_argument('--dataset', type=str, default='query_workload_train.csv',
                        help="Target CSV query log to learn from")
    parser.add_argument('--batch', type=int, default=256, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for the Adam optimizer")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Number of neurons in the hidden layers")

    args = parser.parse_args()
    train(args)
