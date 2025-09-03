import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention_weights(attention_weights, src_tokens, tgt_tokens, layer=0, head=0):
    """
    Attention 가중치 시각화
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights[layer][head].cpu().numpy(),
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='Blues'
    )
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Source Tokens')
    plt.ylabel('Target Tokens')
    plt.show()

def plot_training_curves(train_losses, val_losses):
    """
    학습 곡선 시각화
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
