import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
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
    print(f"\n[EXPORT] Exporting optimal weights to '{filename}'")
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


def train_model(params, dataset, device, trial=None):
    BATCH_SIZE = params['batch_size']
    LEARNING_RATE = params['lr']
    HIDDEN_DIM = params['hidden_dim']
    EPOCHS = params['epochs']

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    model = CDFNet(hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_max_err = float('inf')

    for epoch in range(EPOCHS):
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
        max_err, _ = calculate_max_error(model, test_loader, device)
        scheduler.step(avg_loss)

        if max_err < best_max_err:
            best_max_err = max_err

        # Report to Optuna for early stopping/pruning
        if trial:
            trial.report(best_max_err, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return model, best_max_err


def objective(trial, dataset, device, max_epochs, dataset_filename):
    # Adapt the search space based on the dataset size/type
    if 'network' in dataset_filename.lower() or 'bbc' in dataset_filename.lower():
        params = {
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'hidden_dim': trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            'epochs': max_epochs // 2
        }
    else:
        params = {
            'batch_size': trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096]),
            'lr': trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            'hidden_dim': trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            'epochs': max_epochs // 2
        }

    _, best_max_err = train_model(params, dataset, device, trial)
    return best_max_err


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for the Learned CDF Oracle")
    parser.add_argument('--dataset', type=str, default='query_workload_train.csv',
                        help="Target CSV file to learn from")
    parser.add_argument('--trials', type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument('--epochs', type=int, default=30, help="Max epochs for the final retraining phase")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 55)
    print(f"[TUNING] Initializing Optuna on Device: {str(device).upper()}")
    print("=" * 55)
    print(f"  -> Dataset: {args.dataset}")
    print(f"  -> Trials:  {args.trials}")
    print("-" * 55)

    full_dataset = CDFDataset(filepath=args.dataset, num_items=500000)

    print(f"\n[INFO] Starting Hyperparameter Search ({args.trials} Trials)...")

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, full_dataset, device, args.epochs, args.dataset),
                   n_trials=args.trials)

    print("\n" + "=" * 55)
    print("[SUCCESS] Tuning Complete!")
    print(f"          -> Best Max Error: {study.best_value:.5f}")
    print(f"          -> Best Params:    {study.best_params}")
    print("=" * 55)

    print("\n[INFO] Retraining Final Model with Best Parameters...")
    final_params = study.best_params
    final_params['epochs'] = args.epochs

    final_model, final_err = train_model(final_params, full_dataset, device, trial=None)

    print(f"\n[SUCCESS] Final Model Max Error: {final_err:.5f}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "weights.txt")

    export_weights_to_cpp(final_model, full_dataset.max_key, filename=weights_path)


if __name__ == "__main__":
    main()
