import torch
import matplotlib.pyplot as plt


def plot_attention(sequence, attention_weights):
    """
    Visualize attention by averaging across heads and query positions.
    """

    # shape: (heads, seq_len, seq_len)
    attn = attention_weights.squeeze().cpu().detach()

    # average across heads
    attn = attn.mean(dim=0)

    # average across query positions → importance per position
    attn = attn.mean(dim=0).numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(attn)
    plt.title("Attention over RNA sequence (Self-Attention)")
    plt.xlabel("Position")
    plt.ylabel("Attention weight")
    plt.show()