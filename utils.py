# utils.py
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np

# Fusion entropique simple
def entropy_weighted_fusion(features):
    # features : liste de tensors (B, C, L) ou (B, C, H, W)
    fused = features[0]
    for f in features[1:]:
        fused = (fused + f)/2
    return fused

# Génération matrice de confusion
def compute_confusion(y_true, y_pred, num_classes=61):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
