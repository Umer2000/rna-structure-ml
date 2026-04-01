import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import RNADataset, collate_fn
from model import RNAModel
from visualize import plot_attention


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/pseudo_train.csv"
BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5


# -----------------------------
# Data loading
# -----------------------------
def load_data(path):
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def create_dataloader(df, batch_size):
    """Create DataLoader with padding and masking support."""
    dataset = RNADataset(df)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


# -----------------------------
# Training logic
# -----------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Run one training epoch with efficient masking and gradient control."""
    model.train()
    total_loss = 0

    for x, y, lengths in loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)

        # --- Efficient mask creation ---
        seq_len = y.size(1)
        lengths_tensor = torch.tensor(lengths, device=device)

        mask = torch.arange(seq_len, device=device)[None, :] < lengths_tensor[:, None]
        mask = mask.unsqueeze(-1).float()

        # --- Compute masked loss ---
        loss = loss_fn(preds * mask, y * mask)

        optimizer.zero_grad()
        loss.backward()

        # --- Gradient clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# Main training loop
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = load_data(DATA_PATH)
    loader = create_dataloader(df, BATCH_SIZE)

    # Model setup
    model = RNAModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss:.4f}")

    # -----------------------------
    # Save model ✅ ADDED HERE
    # -----------------------------
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

    # -----------------------------
    # Attention Visualization
    # -----------------------------
    model.eval()

    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)

            preds, attn = model(x, return_attention=True)

            print("Sample sequence indices:", x[0][:50])

            plot_attention(x[0], attn[0].detach())
            break


if __name__ == "__main__":
    main()