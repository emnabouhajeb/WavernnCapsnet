import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


#########################################
# BASIC METRICS
#########################################

def compute_accuracy(pred, target):
    """
    Computes simple classification accuracy.
    pred, target: Tensors of shape [batch]
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    correct = (pred == target).sum()
    return correct / len(target)


def compute_per(pred, target):
    """
    PER = Phoneme Error Rate = 1 - accuracy
    """
    acc = compute_accuracy(pred, target)
    return 1.0 - acc


#########################################
# CONFUSION MATRIX UTILITIES
#########################################

def phoneme_confusion_matrix(pred, target, phoneme_list):
    """
    Generate normalized confusion matrix.
    """
    pred = np.array(pred)
    target = np.array(target)

    cm = confusion_matrix(target, pred, labels=range(len(phoneme_list)))

    # Normalize rows → easier to read confusable phonemes
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    return cm, cm_norm


def save_confusion_matrix(cm_norm, phoneme_list, title, save_path):
    """
    Save a high-resolution confusion matrix plot.
    """
    fig = plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation='nearest', aspect='auto')
    plt.title(title, fontsize=14)
    plt.colorbar(label="Normalized Error Density")

    ticks = np.arange(len(phoneme_list))
    plt.xticks(ticks, phoneme_list, rotation=90, fontsize=6)
    plt.yticks(ticks, phoneme_list, fontsize=6)

    plt.xlabel("Predicted Phoneme")
    plt.ylabel("True Phoneme")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"✓ Confusion matrix saved: {save_path}")
