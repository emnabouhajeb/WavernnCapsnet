"""
Script de validation de l'architecture complète
Teste chaque composant individuellement avant l'entraînement complet
"""

import torch
import os
from dataset import MacroClassDataset, TIMITTestDataset, pad_collate_wavernn, pad_collate_capsnet
from torch.utils.data import DataLoader
from models import (
    WaveRNNClassifier,
    WaveRNNFeatureExtractor,
    CapsNetWithTransferLearning,
    CapsNetWithFusionAndTransfer,
    DEVICE
)

print("=" * 60)
print("ARCHITECTURE VALIDATION SCRIPT")
print("=" * 60)

# ==========================================
# TEST 1: DATASETS
# ==========================================
print("\n[TEST 1] Validating Datasets...")

# Test MacroClassDataset
print("\n  1.1 MacroClassDataset (Vowels)...")
try:
    dataset_vowels = MacroClassDataset(macro_class='Vowels', max_samples=10)
    assert len(dataset_vowels) > 0, "No samples found!"
    sample = dataset_vowels[0]
    assert 'waveform' in sample
    assert 'phoneme' in sample
    print(f"      ✓ Loaded {len(dataset_vowels)} samples")
    print(f"      ✓ Sample shape: {sample['waveform'].shape}")
    print(f"      ✓ Phoneme: {sample['phoneme_name']} (idx={sample['phoneme']})")
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    exit(1)

# Test TIMITTestDataset
print("\n  1.2 TIMITTestDataset...")
try:
    dataset_timit = TIMITTestDataset(split='TEST', max_samples=10)
    assert len(dataset_timit) > 0, "No samples found!"
    sample = dataset_timit[0]
    print(f"      ✓ Loaded {len(dataset_timit)} samples")
    print(f"      ✓ Total phonemes: {dataset_timit.num_phonemes}")
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    exit(1)

# Test DataLoader
print("\n  1.3 DataLoader with collate...")
try:
    loader = DataLoader(dataset_vowels, batch_size=4, collate_fn=pad_collate_wavernn)
    batch = next(iter(loader))
    print(f"      ✓ Batch waveform: {batch['waveform'].shape}")
    print(f"      ✓ Batch phoneme: {batch['phoneme'].shape}")
    assert batch['waveform'].dim() == 3  # [batch, 1, time]
    assert batch['phoneme'].dim() == 1  # [batch]
except Exception as e:
    print(f"      ✗ FAILED: {e}")
    exit(1)

print("\n✓ TEST 1 PASSED: Datasets working correctly")

# ==========================================
# TEST 2: WAVERNN CLASSIFIER
# ==========================================
print("\n[TEST 2] Validating WaveRNN Classifier...")

try:
    model = WaveRNNClassifier(hidden_size=128, num_classes=20).to(DEVICE)

    # Test forward pass
    x = torch.randn(4, 1, 16000).to(DEVICE)  # [batch, 1, time]
    logits = model(x)
    print(f"  ✓ Forward pass: {x.shape} → {logits.shape}")
    assert logits.shape == (4, 20), f"Expected (4, 20), got {logits.shape}"

    # Test extract_features
    features = model.extract_features(x)
    print(f"  ✓ Extract features: {x.shape} → {features.shape}")
    assert features.dim() == 3, "Features should be 3D [batch, time, hidden]"
    assert features.shape[2] == 256, f"Expected hidden_size*2=256, got {features.shape[2]}"

    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n✓ TEST 2 PASSED: WaveRNN Classifier working correctly")

# ==========================================
# TEST 3: WAVERNN PRE-TRAINED (si disponibles)
# ==========================================
print("\n[TEST 3] Checking Pre-trained WaveRNN models...")

WAVERNN_PATHS = {
    'Vowels': 'checkpoints/wavernn_Vowels.pth',
    'Stops': 'checkpoints/wavernn_Stops.pth',
    'Fricatives': 'checkpoints/wavernn_Fricatives.pth',
    'Nasals': 'checkpoints/wavernn_Nasals.pth',
    'Affricates': 'checkpoints/wavernn_Affricates.pth',
    'Semivowels': 'checkpoints/wavernn_Semivowels.pth',
    'Others': 'checkpoints/wavernn_Others.pth'
}

available_paths = {}
wavernn_configs = {}

for name, path in WAVERNN_PATHS.items():
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                config = checkpoint['config']
                available_paths[name] = path
                wavernn_configs[name] = config
                print(f"  ✓ {name}: hidden_size={config['hidden_size']}, num_classes={config['num_classes']}")
            else:
                print(f"  ⚠ {name}: checkpoint found but no config")
        except Exception as e:
            print(f"  ✗ {name}: Error loading - {e}")
    else:
        print(f"  ⚠ {name}: Not found")

if len(available_paths) > 0:
    print(f"\n  ✓ Found {len(available_paths)}/7 pre-trained models")

    # Test WaveRNNFeatureExtractor
    print("\n  Testing WaveRNNFeatureExtractor...")
    try:
        extractor = WaveRNNFeatureExtractor(available_paths, wavernn_configs).to(DEVICE)
        x = torch.randn(2, 1, 16000).to(DEVICE)
        features = extractor(x)
        print(f"    ✓ Feature extraction: {x.shape} → {features.shape}")
        assert features.shape == (2, 100, 256), f"Expected (2, 100, 256), got {features.shape}"
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
else:
    print("\n  ⚠ No pre-trained models found. Will train from scratch.")
    print("    To use transfer learning, train WaveRNN models first:")
    print("    python wavernn_pretrain.py --macroclass Vowels")

print("\n✓ TEST 3 PASSED")

# ==========================================
# TEST 4: CAPSNET
# ==========================================
print("\n[TEST 4] Validating CapsNet...")

try:
    if len(available_paths) > 0:
        model = CapsNetWithTransferLearning(
            num_classes=61,
            wavernn_paths=available_paths,
            wavernn_configs=wavernn_configs
        ).to(DEVICE)
        print("  ✓ CapsNet with transfer learning initialized")
    else:
        model = CapsNetWithTransferLearning(
            num_classes=61,
            wavernn_paths=None,
            wavernn_configs=None
        ).to(DEVICE)
        print("  ✓ CapsNet without transfer learning initialized")

    # Test forward pass
    x = torch.randn(2, 1, 16000).to(DEVICE)
    output = model(x)
    print(f"  ✓ Forward pass: {x.shape} → {output.shape}")
    assert output.shape == (2, 61), f"Expected (2, 61), got {output.shape}"

    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n✓ TEST 4 PASSED: CapsNet working correctly")

# ==========================================
# TEST 5: FULL MODEL WITH FUSION
# ==========================================
print("\n[TEST 5] Validating Full Model with Fusion...")

try:
    if len(available_paths) > 0:
        model = CapsNetWithFusionAndTransfer(
            num_classes=61,
            wavernn_paths=available_paths,
            wavernn_configs=wavernn_configs
        ).to(DEVICE)
    else:
        model = CapsNetWithFusionAndTransfer(
            num_classes=61,
            wavernn_paths=None,
            wavernn_configs=None
        ).to(DEVICE)

    print("  ✓ Full model initialized")

    # Test sans fusion
    x = torch.randn(2, 1, 16000).to(DEVICE)
    probs_no_fusion = model(x, apply_fusion=False)
    print(f"  ✓ Forward (no fusion): {x.shape} → {probs_no_fusion.shape}")
    assert probs_no_fusion.shape == (2, 61)
    assert torch.allclose(probs_no_fusion.sum(dim=1), torch.ones(2).to(DEVICE), atol=1e-5), \
        "Probabilities should sum to 1"

    # Test avec fusion
    probs_fusion = model(x, apply_fusion=True)
    print(f"  ✓ Forward (with fusion): {x.shape} → {probs_fusion.shape}")
    assert probs_fusion.shape == (2, 61)
    assert torch.allclose(probs_fusion.sum(dim=1), torch.ones(2).to(DEVICE), atol=1e-5), \
        "Probabilities should sum to 1"

    # Vérifier que fusion change les prédictions
    diff = (probs_no_fusion - probs_fusion).abs().max().item()
    print(f"  ✓ Max difference between fusion/no-fusion: {diff:.6f}")

    print(f"  ✓ Total model parameters: {sum(p.numel() for p in model.parameters()):,}")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n✓ TEST 5 PASSED: Full model with fusion working correctly")

# ==========================================
# TEST 6: BACKWARD PASS (GRADIENT FLOW)
# ==========================================
print("\n[TEST 6] Validating Gradient Flow...")

try:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(2, 1, 16000).to(DEVICE)
    y = torch.tensor([0, 1], dtype=torch.long).to(DEVICE)

    # Forward + backward
    optimizer.zero_grad()
    probs = model(x, apply_fusion=True)
    loss = torch.nn.functional.cross_entropy(probs, y)
    loss.backward()

    # Vérifier les gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm == 0:
                print(f"    ⚠ Warning: Zero gradient for {name}")

    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Parameters with gradients: {len(grad_norms)}")
    print(f"  ✓ Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")

    optimizer.step()
    print(f"  ✓ Optimizer step successful")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n✓ TEST 6 PASSED: Gradient flow working correctly")

# ==========================================
# TEST 7: MINI TRAINING LOOP
# ==========================================
print("\n[TEST 7] Testing Mini Training Loop...")

try:
    model.train()
    loader = DataLoader(dataset_timit, batch_size=2, collate_fn=pad_collate_capsnet)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("  Running 3 mini-batches...")
    for i, batch in enumerate(loader):
        if i >= 3:
            break

        x = batch['waveform'].to(DEVICE)
        y = batch['phoneme'].to(DEVICE)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        optimizer.zero_grad()
        probs = model(x, apply_fusion=False)
        loss = torch.nn.functional.cross_entropy(probs, y)
        loss.backward()
        optimizer.step()

        acc = (probs.argmax(dim=-1) == y).float().mean().item() * 100
        print(f"    Batch {i + 1}: loss={loss.item():.4f}, acc={acc:.1f}%")

    print("  ✓ Mini training loop successful")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n✓ TEST 7 PASSED: Training loop working correctly")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("ALL VALIDATION TESTS PASSED! ✓")
print("=" * 60)
print("\nArchitecture is ready for training!")
print("\nNext steps:")
print("  1. Train WaveRNN models (if not done yet):")
print("     python wavernn_pretrain.py --macroclass Vowels")
print("     python wavernn_pretrain.py --macroclass Stops")
print("     etc...")
print("\n  2. Train full CapsNet model:")
print("     python capsnet_train.py")
print("\n" + "=" * 60)