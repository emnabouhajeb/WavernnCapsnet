import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MacroClassDataset, pad_collate_wavernn
from tqdm import tqdm
import os
import argparse

# ===========================
# CONFIG
# ===========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
EPOCHS = 120
MAX_SAMPLES = None  # Utiliser toutes les données


# ===========================
# WAVE RNN pour CLASSIFICATION (pas génération)
# ===========================
class WaveRNNClassifier(nn.Module):
    """
    WaveRNN adapté pour la CLASSIFICATION de macro-classes
    Architecture inspirée de l'article Section 3.1
    """

    def __init__(self, hidden_size=128, num_classes=1):
        super().__init__()
        # Convolutions dilatées pour capturer les patterns temporels
        # Dilations exponentielles: 1, 2, 4, 8, 16 (couvre ~1000 samples)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=3, dilation=2 ** i, padding=2 ** i)
            for i in range(5)
        ])
        self.conv_pool = nn.Conv1d(64 * 5, 128, kernel_size=1)

        # GRU bidirectionnel pour modélisation temporelle
        self.rnn1 = nn.GRU(128, hidden_size, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)

        # Couche de classification
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [batch, 1, time]

        # Convolutions dilatées pour capturer multi-échelles
        conv_outputs = []
        for conv in self.conv_layers:
            out = F.relu(conv(x))
            conv_outputs.append(out)

        # Concaténer toutes les sorties
        x = torch.cat(conv_outputs, dim=1)  # [batch, 64*5, time]
        x = F.relu(self.conv_pool(x))  # [batch, 128, time]

        # Pooling adaptatif pour avoir une longueur fixe
        x = F.adaptive_avg_pool1d(x, 100)  # [batch, 128, 100]

        x = x.transpose(1, 2)  # [batch, 100, 128]

        # GRU bidirectionnel
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)

        # Global pooling sur la dimension temporelle
        x = x.mean(dim=1)  # [batch, hidden_size*2]

        # Classification
        x = self.fc(x)  # [batch, num_classes]
        return x

    def extract_features(self, x):
        """Extrait les features RNN pour transfer learning"""
        # Même pipeline jusqu'avant la couche fc
        conv_outputs = []
        for conv in self.conv_layers:
            out = F.relu(conv(x))
            conv_outputs.append(out)

        x = torch.cat(conv_outputs, dim=1)
        x = F.relu(self.conv_pool(x))
        x = F.adaptive_avg_pool1d(x, 100)
        x = x.transpose(1, 2)

        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)

        return x  # [batch, 100, hidden_size*2]


# ===========================
# MAIN TRAIN FUNCTION
# ===========================
def train_macroclass(class_name):
    print(f"\n{'=' * 50}\nWaveRNN Macro-Class Classification: {class_name}\n{'=' * 50}")

    dataset = MacroClassDataset(macro_class=class_name, max_samples=MAX_SAMPLES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=pad_collate_wavernn, num_workers=0)

    # Le nombre de phonèmes dans cette macro-classe
    num_phonemes = len(dataset.phonemes_in_class)
    print(f"Training for {num_phonemes} phonemes in {class_name}")

    model = WaveRNNClassifier(hidden_size=128, num_classes=num_phonemes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("checkpoints", exist_ok=True)
    output_path = f"checkpoints/wavernn_{class_name}.pth"

    best_loss = float('inf')
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            x = batch['waveform'].to(DEVICE)  # [batch, 1, time]
            y = batch['phoneme'].to(DEVICE)  # [batch]

            if x.size(-1) < 100:
                continue

            optimizer.zero_grad()

            # Forward pass
            logits = model(x)  # [batch, num_phonemes]
            loss = F.cross_entropy(logits, y)

            if torch.isnan(loss):
                print("⚠ NaN loss, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Métriques
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

            acc = 100.0 * correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

        avg_loss = total_loss / len(loader)
        avg_acc = 100.0 * correct / total if total > 0 else 0
        scheduler.step()

        print(
            f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Acc = {avg_acc:.2f}%, LR = {optimizer.param_groups[0]['lr']:.6f}")

        # Sauvegarder si meilleure accuracy
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'hidden_size': 128,
                    'num_classes': num_phonemes,
                    'macro_class': class_name
                },
                'accuracy': avg_acc,
                'loss': avg_loss,
                'phonemes': dataset.phonemes_in_class
            }, output_path)
            print(f"✓ New best! Acc={avg_acc:.2f}% Saved: {output_path}")

    print(f"[DONE] {class_name} → Best Acc: {best_acc:.2f}%")


# ===========================
# ARGPARSE
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--macroclass', type=str, required=True,
                        choices=['Vowels', 'Stops', 'Fricatives', 'Nasals',
                                 'Affricates', 'Semivowels', 'Others'])
    args = parser.parse_args()
    train_macroclass(args.macroclass)