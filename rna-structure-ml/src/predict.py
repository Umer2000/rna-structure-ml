import os
import argparse
import torch
import pandas as pd
from Bio import SeqIO

from model import RNAModel
from dataset import encode_sequence


# -----------------------------
# Load trained model
# -----------------------------
def load_model(model_path, device):
    model = RNAModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# -----------------------------
# Predict one sequence
# -----------------------------
def predict_sequence(model, sequence, device):
    encoded = encode_sequence(sequence)

    x = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)

    return preds.squeeze(0).cpu().numpy()


# -----------------------------
# Predict from FASTA
# -----------------------------
def predict_from_fasta(model, fasta_path, device):
    results = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)

        coords = predict_sequence(model, sequence, device)

        for i, (x, y, z) in enumerate(coords):
            results.append({
                "sequence_id": seq_id,
                "position": i + 1,
                "residue": sequence[i],
                "x": float(x),
                "y": float(y),
                "z": float(z)
            })

    return pd.DataFrame(results)


# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="RNA Structure Prediction Tool")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input FASTA file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to output CSV file"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Path to trained model"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.model, device)

    # Run prediction
    df = predict_from_fasta(model, args.input, device)

    # Save results
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()