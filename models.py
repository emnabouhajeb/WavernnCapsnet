"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class WaveRNNClassifier(nn.Module):


    def __init__(self, hidden_size=128, num_classes=1):
        super().__init__()

        # 9-layer dilated convolutions (d=2^(i*0.8)) comme l'article
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=3,
                      dilation=int(2 ** (i * 0.8)),
                      padding=int(2 ** (i * 0.8)))
            for i in range(9)  # 9 layers exact
        ])
        self.conv_pool = nn.Conv1d(64 * 9, 128, kernel_size=1)

        # Bidirectional GRUs (Eq. 1-3)
        self.rnn1 = nn.GRU(128, hidden_size, batch_first=True, bidirectional=True)
        self.rnn2 = nn.GRU(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        conv_outputs = [F.relu(conv(x)) for conv in self.conv_layers]
        x = torch.cat(conv_outputs, dim=1)
        x = F.relu(self.conv_pool(x))
        x = F.adaptive_avg_pool1d(x, 100)
        x = x.transpose(1, 2)
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        return self.fc(x.mean(dim=1))

    def extract_features(self, x):
        conv_outputs = [F.relu(conv(x)) for conv in self.conv_layers]
        x = torch.cat(conv_outputs, dim=1)
        x = F.relu(self.conv_pool(x))
        x = F.adaptive_avg_pool1d(x, 100)
        x = x.transpose(1, 2)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        return x  # [B, 100, 256]


def load_wavernn_checkpoint(path, device='cpu'):
    if not os.path.exists(path):
        return None

    try:
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        config = ckpt.get('config', {})

        hidden_size = config.get('hidden_size', 128)
        num_classes = config.get('num_classes', 1)
        if 'fc.weight' in state:
            num_classes = state['fc.weight'].shape[0]

        model = WaveRNNClassifier(hidden_size, num_classes)
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        return model
    except Exception as e:
        print(f"✗ Failed to load {path}: {e}")
        return None


class WaveRNNFeatureExtractor(nn.Module):
    """Stage 1: Extract features from 7 WaveRNN macro-classes"""

    def __init__(self, checkpoint_dir='checkpoints', device=DEVICE):
        super().__init__()
        self.device = device
        self.extractors = nn.ModuleDict()

        macro_classes = ['Vowels', 'Stops', 'Fricatives', 'Nasals',
                         'Affricates', 'Semivowels', 'Others']

        for cls in macro_classes:
            path = os.path.join(checkpoint_dir, f'wavernn_{cls}.pth')
            model = load_wavernn_checkpoint(path, device)
            if model is not None:
                self.extractors[cls] = model

        print(f"✓ WaveRNN Feature Extractor: {len(self.extractors)}/7 models loaded")

        # Aggregator learnable
        if len(self.extractors) > 0:
            self.aggregator = nn.Sequential(
                nn.Linear(256 * len(self.extractors), 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256)
            )

    def forward(self, x):
        if len(self.extractors) == 0:
            return None

        features = []
        with torch.no_grad():
            for extractor in self.extractors.values():
                feat = extractor.extract_features(x.to(self.device))
                features.append(feat)

        # Concatenate all macro-class features
        stacked = torch.stack(features, dim=1)  # [B, 7, 100, 256]
        B, N, T, C = stacked.shape
        concatenated = stacked.view(B, T, N * C)  # [B, 100, 1792]

        # Learnable aggregation
        aggregated = self.aggregator(concatenated)  # [B, 100, 256]

        return aggregated.detach()


# ==========================================
# CAPSNET
# ==========================================
def squash(v, dim=-1, epsilon=1e-7):
    norm_sq = (v ** 2).sum(dim=dim, keepdim=True).clamp(min=epsilon)
    scale = norm_sq / (1.0 + norm_sq)
    return scale * v / torch.sqrt(norm_sq + epsilon)


class DynamicRoutingCapsuleLayer(nn.Module):

    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_iterations=4):
        super().__init__()
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.num_iterations = num_iterations

        # Transformation matrices W_ij
        self.W = nn.Parameter(
            torch.randn(1, in_capsules, out_capsules, out_dim, in_dim) * 0.01
        )

    def forward(self, u):
        """
        u: [B, in_capsules, in_dim]
        Returns: [B, out_capsules, out_dim]
        """
        B = u.size(0)

        # Prediction vectors
        u_expand = u.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, u_expand).squeeze(-1)

        # Initialize routing logits
        b_ij = torch.zeros(B, self.in_capsules, self.out_capsules, 1,
                           device=u.device, dtype=u.dtype)

        # Algorithm 1: 4 iterations
        for iteration in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1)
            v_j = squash(s_j, dim=-1)

            if iteration < self.num_iterations - 1:
                agreement = (u_hat * v_j.unsqueeze(1)).sum(dim=-1, keepdim=True)
                b_ij = b_ij + agreement

        return v_j


class HierarchicalCapsNet(nn.Module):

    def __init__(self, num_phonemes=61):
        super().__init__()
        self.num_phonemes = num_phonemes

        # Feature extraction (Figure 4)
        self.conv1 = nn.Conv1d(1, 256, kernel_size=19, stride=1, padding=9)
        self.bn1 = nn.BatchNorm1d(256)

        self.depthwise = nn.Conv1d(256, 512, kernel_size=9, groups=8, padding=4)
        self.bn2 = nn.BatchNorm1d(512)

        # Primary Capsules (128 capsules × 8D)
        self.primary_caps_conv = nn.Conv1d(512, 128 * 8, kernel_size=9, stride=2, padding=4)

        # Phoneme Capsules (61 capsules × 16D) via Dynamic Routing
        self.phoneme_caps = DynamicRoutingCapsuleLayer(
            in_capsules=128,
            in_dim=8,
            out_capsules=num_phonemes,
            out_dim=16,
            num_iterations=4  # Exactement 4 comme l'article
        )

        # Decoder pour Reconstruction (Section 4.6, Eq. 11)
        self.decoder = nn.Sequential(
            nn.Linear(num_phonemes * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512)  # Reconstruit features
        )

        self.target_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )

    def forward(self, x, reconstruct=False):
        B = x.size(0)

        # Feature extraction
        feat1 = F.relu(self.bn1(self.conv1(x)))
        feat2 = F.relu(self.bn2(self.depthwise(feat1)))

        # Primary capsules
        primary = self.primary_caps_conv(feat2)  # [B, 128*8, T/2]
        primary = primary.view(B, 128, 8, -1)
        primary = primary.permute(0, 1, 3, 2)
        primary = squash(primary, dim=-1)
        primary = primary.mean(dim=2)  # [B, 128, 8]

        # Dynamic routing → Phoneme capsules
        phoneme_caps = self.phoneme_caps(primary)  # [B, 61, 16]

        if reconstruct:
            # Reconstruction
            caps_flat = phoneme_caps.view(B, -1)
            reconstruction = self.decoder(caps_flat)

            # Target: pooled CNN features
            target = F.adaptive_avg_pool1d(feat2, 1).squeeze(-1)
            target = self.target_projection(target)

            return phoneme_caps, reconstruction, target

        return phoneme_caps


# ==========================================
# TRANSFER LEARNING
# ==========================================
class AdaptiveAlignmentGate(nn.Module):

    def __init__(self, wavernn_dim=256, caps_dim=16, hidden_dim=128):
        super().__init__()
        self.query = nn.Linear(wavernn_dim, hidden_dim)
        self.key = nn.Linear(caps_dim, hidden_dim)
        self.value = nn.Linear(caps_dim, caps_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, h_wavernn, c_caps):
        """
        h_wavernn: [B, T_wav, 256]
        c_caps: [B, num_caps, 16]
        """
        Q = self.query(h_wavernn)  # [B, T_wav, 128]
        K = self.key(c_caps)  # [B, num_caps, 128]
        V = self.value(c_caps)

        # Scaled dot-product attention (Eq. 7)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        alpha = F.softmax(scores, dim=-1)

        # Weighted aggregation
        aligned = torch.matmul(alpha.transpose(-2, -1), Q)

        # Residual
        return V + aligned[:, :, :16]


# ==========================================
# PROBABILISTIC FUSION
# ==========================================
class EntropyWeightedFusion(nn.Module):

    def __init__(self, num_classes=61, dropout_p=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # Similarity matrix S (learnable)
        self.S = nn.Parameter(
            torch.eye(num_classes) * 0.9 + torch.randn(num_classes, num_classes) * 0.02
        )

    def forward(self, v_j, apply_fusion=True):
        """
        v_j: [B, num_classes, capsule_dim]
        Returns: [B, num_classes]
        """
        # Capsule norms = probabilities
        norms = torch.norm(v_j, dim=-1)
        probs = F.softmax(norms, dim=-1)

        if not apply_fusion:
            return probs

        # Eq. 8: Reliability via entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        max_entropy = np.log(self.num_classes)
        reliability = 1.0 - (entropy / max_entropy)
        reliability = reliability.unsqueeze(-1)

        # Contextual relationships
        contextual = probs @ self.S

        # Monte Carlo dropout
        if self.training:
            mask = torch.bernoulli(
                torch.ones_like(contextual) * (1 - self.dropout_p)
            )
            contextual = contextual * mask

        # Eq. 9: Final fusion
        weighted = reliability * contextual
        fused = F.softmax(weighted, dim=-1)

        return fused



class OptimalSpeechRecognitionSystem(nn.Module):
    """

    """

    def __init__(self, num_phonemes=61, checkpoint_dir='checkpoints'):
        super().__init__()
        self.num_phonemes = num_phonemes

        # Stage 1: WaveRNN extractors (frozen)
        self.wavernn_extractor = WaveRNNFeatureExtractor(checkpoint_dir, DEVICE)
        self.use_wavernn = (len(self.wavernn_extractor.extractors) > 0)

        # Stage 2: CapsNet
        self.capsnet = HierarchicalCapsNet(num_phonemes)

        # Transfer learning alignment
        if self.use_wavernn:
            self.alignment_gate = AdaptiveAlignmentGate(
                wavernn_dim=256,
                caps_dim=16,
                hidden_dim=128
            )

        # Probabilistic fusion
        self.fusion = EntropyWeightedFusion(num_phonemes, dropout_p=0.2)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_phonemes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_phonemes)
        )

    def forward(self, x, apply_fusion=False, reconstruct=False):
        B = x.size(0)

        # Stage 1: WaveRNN features (frozen)
        if self.use_wavernn:
            with torch.no_grad():
                wavernn_feats = self.wavernn_extractor(x)
        else:
            wavernn_feats = None

        # Stage 2: CapsNet
        if reconstruct:
            caps_output, reconstruction, target = self.capsnet(x, reconstruct=True)
        else:
            caps_output = self.capsnet(x)
            reconstruction = None
            target = None

        # Transfer learning (Eq. 7)
        if self.use_wavernn and wavernn_feats is not None:
            caps_output = self.alignment_gate(wavernn_feats, caps_output)

        # Probabilistic fusion (Eq. 8-9)
        probs = self.fusion(caps_output, apply_fusion=apply_fusion)

        # Final classification
        logits = self.classifier(probs)

        if reconstruct:
            return logits, reconstruction, target
        return logits

    def margin_loss(self, v_c, labels, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        """Eq. 10: Margin loss pour capsules"""
        B = labels.size(0)
        v_norm = torch.norm(v_c, dim=-1)
        T_c = F.one_hot(labels, self.num_phonemes).float()

        loss_present = T_c * F.relu(m_plus - v_norm) ** 2
        loss_absent = lambda_ * (1 - T_c) * F.relu(v_norm - m_minus) ** 2

        return (loss_present + loss_absent).sum(dim=-1).mean()


# ==========================================
# PREPROCESSING
# ==========================================
def mu_law_companding(x, mu=255):
    """Eq. 4: ITU-T G.711 companding"""
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    return sign * torch.log1p(mu * x_abs) / np.log(1 + mu)


def preprocess_waveform(waveform, apply_vad=True, apply_companding=True):

    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    # VAD (energy-based)
    if apply_vad and len(waveform) > 400:
        energy = waveform ** 2
        threshold = energy.mean() * 0.1
        mask = energy > threshold
        if mask.sum() > 0:
            waveform = waveform[mask]

    # Companding (Eq. 4)
    if apply_companding:
        waveform = mu_law_companding(waveform)

    # Normalization
    waveform = waveform / (waveform.abs().max() + 1e-8)

    return waveform


if __name__ == "__main__":
    print("=" * 60)
    print()
    print("=" * 60)

    model = OptimalSpeechRecognitionSystem(
        num_phonemes=61,
        checkpoint_dir='checkpoints'
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"\n✓ Total parameters: {n_params:.1f}M")
    print(f"✓ Trainable: {trainable:.1f}M")
    print(f"✓ Device: {DEVICE}")

    # Test forward
    x = torch.randn(8, 1, 16000).to(DEVICE)
    y = torch.randint(0, 61, (8,)).to(DEVICE)

    logits, recon, target = model(x, apply_fusion=True, reconstruct=True)

    print(f"\n✓ Forward pass OK: {logits.shape}")
    print(f"✓ Reconstruction: {recon.shape}")
    print(f"✓ Ready for training!")