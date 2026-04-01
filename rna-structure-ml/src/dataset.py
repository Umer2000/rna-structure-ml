import ast
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# -----------------------------
# Configuration
# -----------------------------
# NOTE: 0 is reserved for padding
NUCLEOTIDE_MAP = {
    "A": 1,
    "C": 2,
    "G": 3,
    "U": 4
}

MAX_LEN = 200  # reduced for better attention learning


# -----------------------------
# Utility functions
# -----------------------------
def encode_sequence(sequence):
    """Convert RNA sequence string into integer indices."""
    return [NUCLEOTIDE_MAP.get(base, 0) for base in sequence]


def parse_coordinates(coord_str):
    """Convert stored string representation of coordinates back to list."""
    return ast.literal_eval(coord_str)


# -----------------------------
# Dataset class
# -----------------------------
class RNADataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = dataframe["sequence"].values
        self.coords = dataframe["coords"].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        coord_str = self.coords[idx]

        # Encode sequence
        encoded_seq = encode_sequence(sequence)[:MAX_LEN]

        # Parse and normalize coordinates
        coords = parse_coordinates(coord_str)[:MAX_LEN]
        coords = torch.tensor(coords, dtype=torch.float) / 10.0

        return (
            torch.tensor(encoded_seq, dtype=torch.long),
            coords
        )


# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    sequences, coords = zip(*batch)

    lengths = [len(seq) for seq in sequences]

    sequences_padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0  # 0 = padding
    )

    coords_padded = pad_sequence(
        coords,
        batch_first=True,
        padding_value=0.0
    )

    return sequences_padded, coords_padded, lengths