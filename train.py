import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from models.gnn_model import RetailMindGNN, TemporalGraphBuilder

# ── Training Configuration ────────────────────────────────────────────────────
CONFIG = {
    "epochs":       150,
    "lr":           0.001,
    "weight_decay": 1e-4,
    "hidden_dim":   64,
    "heads":        4,
    "horizons":     3,       # 7, 14, 30 day forecasts
    "train_cutoff": "2024-09-30",   # train on data before this
    "val_cutoff":   "2024-12-31",   # validate on this window
    "half_life":    60,             # temporal decay half-life in days
}

HORIZON_DAYS = [7, 14, 30]


# ── Label Builder ─────────────────────────────────────────────────────────────
def build_labels(df: pd.DataFrame,
                 customer_ids: list,
                 spice_ids: list,
                 from_date: str,
                 horizon_days: list) -> torch.Tensor:
    """
    For each (customer, spice, horizon), sum actual quantity ordered
    in the next N days after from_date.
    Returns tensor of shape [n_customers, n_spices, n_horizons]
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    from_dt = pd.to_datetime(from_date)

    c_idx = {c: i for i, c in enumerate(customer_ids)}
    s_idx = {s: i for i, s in enumerate(spice_ids)}

    labels = np.zeros((len(customer_ids), len(spice_ids), len(horizon_days)))

    for h_i, h in enumerate(horizon_days):
        window = df[
            (df["date"] > from_dt) &
            (df["date"] <= from_dt + pd.Timedelta(days=h))
        ]
        for _, row in window.iterrows():
            ci = c_idx.get(row["customer_id"])
            si = s_idx.get(row["spice_id"])
            if ci is not None and si is not None:
                labels[ci, si, h_i] += row["quantity_kg"]

    # Normalise by horizon length so all horizons are on same scale
    for h_i, h in enumerate(horizon_days):
        labels[:, :, h_i] /= h

    return torch.tensor(labels, dtype=torch.float)


# ── Sliding Window Dataset ────────────────────────────────────────────────────
def build_sliding_windows(df: pd.DataFrame,
                           customer_ids: list,
                           spice_ids: list,
                           window_days: int = 90,
                           step_days:   int = 14) -> list:
    """
    Creates multiple (graph, label) pairs by sliding a window
    across the training period. This is the key technique that
    gives the model temporal diversity despite a small dataset.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    start = df["date"].min() + pd.Timedelta(days=window_days)
    end   = pd.to_datetime(CONFIG["train_cutoff"])

    samples = []
    current = start

    while current <= end:
        window_start = current - pd.Timedelta(days=window_days)
        window_df    = df[
            (df["date"] >= window_start) &
            (df["date"] <  current)
        ]

        if len(window_df) < 50:   # skip sparse windows
            current += pd.Timedelta(days=step_days)
            continue

        builder = TemporalGraphBuilder(
            window_df,
            reference_date=current.strftime("%Y-%m-%d")
        )
        graph  = builder.build()
        labels = build_labels(
            df, customer_ids, spice_ids,
            current.strftime("%Y-%m-%d"),
            HORIZON_DAYS
        )
        samples.append((graph, labels))
        current += pd.Timedelta(days=step_days)

    return samples


# ── Training Loop ─────────────────────────────────────────────────────────────
def train():
    print("── RetailMind GNN Training ───────────────────────────────────────")
    df = pd.read_csv("data/orders.csv")

    customer_ids = sorted(df["customer_id"].unique())
    spice_ids    = sorted(df["spice_id"].unique())

    print(f"  Building sliding window dataset...")
    samples = build_sliding_windows(df, customer_ids, spice_ids)
    print(f"  Training samples : {len(samples)}")

    # Train/val split — last 20% of windows for validation
    split     = int(len(samples) * 0.8)
    train_set = samples[:split]
    val_set   = samples[split:]

    model     = RetailMindGNN(
        hidden_dim=CONFIG["hidden_dim"],
        heads=CONFIG["heads"],
        horizons=CONFIG["horizons"]
    )
    optimizer = Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.HuberLoss()   # robust to outliers (luxury hotel spikes)

    best_val_loss = float("inf")
    history       = {"train": [], "val": []}

    print(f"  Training for {CONFIG['epochs']} epochs...\n")

    for epoch in range(1, CONFIG["epochs"] + 1):

        # ── Train ──
        model.train()
        train_loss = 0.0
        for graph, labels in train_set:
            optimizer.zero_grad()
            preds = model(graph)
            loss  = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_set)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph, labels in val_set:
                preds    = model(graph)
                val_loss += criterion(preds, labels).item()
        val_loss /= max(len(val_set), 1)

        scheduler.step()
        history["train"].append(round(train_loss, 4))
        history["val"].append(round(val_loss, 4))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CONFIG['epochs']} "
                  f"| train loss: {train_loss:.4f} "
                  f"| val loss: {val_loss:.4f} "
                  f"{'← best' if val_loss == best_val_loss else ''}")

    # ── Save training history ──
    with open("models/history.json", "w") as f:
        json.dump(history, f)

    print(f"\n✅ Training complete")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Model saved   → models/best_model.pt")
    print(f"   History saved → models/history.json")


if __name__ == "__main__":
    train()