import torch
import torch.nn as nn


class RNAModel(nn.Module):
    def __init__(self):
        """
        RNA sequence model with BiLSTM + masked Multihead Self-Attention.
        """
        super().__init__()

        # -----------------------------
        # Embedding (0 = padding)
        # -----------------------------
        self.embedding = nn.Embedding(
            num_embeddings=5,  # 4 bases + padding
            embedding_dim=32,
            padding_idx=0
        )

        # -----------------------------
        # BiLSTM encoder
        # -----------------------------
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        # -----------------------------
        # Multi-head self-attention
        # -----------------------------
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        # -----------------------------
        # Regularization
        # -----------------------------
        self.dropout = nn.Dropout(0.3)

        # -----------------------------
        # Output layer
        # -----------------------------
        self.fc = nn.Linear(128, 3)

    def forward(self, x, return_attention=False):
        """
        Forward pass with padding-aware attention.
        """

        # -----------------------------
        # Create padding mask (True = ignore)
        # -----------------------------
        padding_mask = (x == 0)

        # -----------------------------
        # Embedding
        # -----------------------------
        x = self.embedding(x)

        # -----------------------------
        # BiLSTM
        # -----------------------------
        lstm_out, _ = self.lstm(x)

        # -----------------------------
        # Self-attention (masked)
        # -----------------------------
        attn_output, attn_weights = self.attention(
            lstm_out,
            lstm_out,
            lstm_out,
            key_padding_mask=padding_mask
        )

        # -----------------------------
        # Output
        # -----------------------------
        x = self.dropout(attn_output)
        output = self.fc(x)

        if return_attention:
            return output, attn_weights

        return output