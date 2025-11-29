import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
import json
from datetime import datetime

from models import OptimalSpeechRecognitionSystem, preprocess_waveform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# CONFIGURATION
# ============================================================
class OptimalConfig:

    # Dataset
    num_phonemes = 61
    timit_root = 'TIMIT'
    sample_rate = 16000

    # Training
    batch_size = 16
    epochs = 100
    num_workers = 8
    pin_memory = True

    # Optimizer
    learning_rate = 3e-4
    weight_decay = 0.01
    betas = (0.9, 0.999)
    eps = 1e-8

    # Scheduler
    scheduler_type = 'cosine'
    warmup_epochs = 5
    min_lr = 1e-6

    # Gradient
    grad_clip = 5.0

    # Loss weights
    margin_loss_weight = 0.5
    recon_loss_weight = 0.1
    label_smoothing = 0.1

    # Regularization
    dropout_rate = 0.3
    spectral_augment = True
    mixup_alpha = 0.2

    # Fusion
    apply_fusion_after_epoch = 30

    # Mixed precision
    use_amp = True

    # Checkpoints
    checkpoint_dir = 'checkpoints'
    save_dir = 'results_optimal'
    save_every = 5
    save_best_only = False

    # Logging
    log_interval = 50

    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


# ============================================================
# CLASS-BALANCED LOSS
# ============================================================
class ClassBalancedCrossEntropy(nn.Module):
    """
    Eq. 5 : class-balanced cross entropy
    """

    def __init__(self, class_counts, label_smoothing=0.1):
        super().__init__()

        weights = 1.0 / torch.sqrt(torch.FloatTensor(class_counts) + 1)
        weights = weights / weights.sum()
        self.register_buffer('weights', weights)

        self.label_smoothing = label_smoothing
        self.num_classes = len(class_counts)

    def forward(self, logits, labels):
        one_hot = F.one_hot(labels, self.num_classes).float()
        smooth_labels = one_hot * (1 - self.label_smoothing) + \
                        (1 - one_hot) * (self.label_smoothing / (self.num_classes - 1))

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)

        weighted_loss = loss * self.weights[labels]

        return weighted_loss.mean()


# ============================================================
# MIXUP
# ============================================================
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# WARMUP + COSINE SCHEDULER
# ============================================================
class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        # Warmup
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs

        else:
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))

        for g in self.optimizer.param_groups:
            g['lr'] = lr

        return lr


# ============================================================
# METRICS
# ============================================================
class Metrics:

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.per_class_correct = {}
        self.per_class_total = {}

    def update(self, predictions, labels):
        preds = predictions.cpu().numpy()
        labs = labels.cpu().numpy()

        self.correct += (preds == labs).sum()
        self.total += len(labs)

        for p, l in zip(preds, labs):
            if l not in self.per_class_total:
                self.per_class_total[l] = 0
                self.per_class_correct[l] = 0
            self.per_class_total[l] += 1
            if p == l:
                self.per_class_correct[l] += 1

    def compute(self):
        acc = 100 * self.correct / max(1, self.total)
        per = 100 - acc

        per_class_acc = {
            cls: 100 * self.per_class_correct[cls] / max(1, self.per_class_total[cls])
            for cls in self.per_class_total
        }

        return {
            'accuracy': acc,
            'PER': per,
            'per_class_accuracy': per_class_acc
        }


# ============================================================
# TRAINER
# ============================================================
class OptimalTrainer:

    def __init__(self, config, train_loader, val_loader, class_counts):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        print("\nInitializing model...")
        self.model = OptimalSpeechRecognitionSystem(
            num_phonemes=config.num_phonemes,
            checkpoint_dir=config.checkpoint_dir
        ).to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            config.warmup_epochs,
            config.epochs,
            config.min_lr,
            config.learning_rate
        )

        self.ce_loss = ClassBalancedCrossEntropy(
            class_counts,
            label_smoothing=config.label_smoothing
        )

        self.scaler = GradScaler() if config.use_amp else None

        self.train_metrics = Metrics()
        self.val_metrics = Metrics()

        self.best_per = float("inf")
        self.best_epoch = 0

        self.history = {
            'train_loss': [],
            'train_per': [],
            'val_loss': [],
            'val_per': [],
            'lr': []
        }

        os.makedirs(config.save_dir, exist_ok=True)

    # ----------------------------------------------------------
    # TRAIN EPOCH
    # ----------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0
        n_batches = 0

        apply_fusion = (epoch >= self.config.apply_fusion_after_epoch)
        use_mixup = (epoch < 50)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            x = batch['waveform'].to(DEVICE)
            y = batch['phoneme'].to(DEVICE)

            if use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, self.config.mixup_alpha)
            else:
                y_a, y_b, lam = y, None, 1

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.use_amp):
                logits, recon, target, caps_output = \
                    self.model.forward_train(x, apply_fusion=apply_fusion)

                if y_b is not None:
                    ce = mixup_criterion(self.ce_loss, logits, y_a, y_b, lam)
                else:
                    ce = self.ce_loss(logits, y_a)

                margin = self.model.margin_loss(caps_output, y_a)
                recon_loss = F.mse_loss(recon, target)

                loss = ce + \
                       self.config.margin_loss_weight * margin + \
                       self.config.recon_loss_weight * recon_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=-1)
            self.train_metrics.update(preds, y)

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "PER": f"{self.train_metrics.compute()['PER']:.2f}%"
            })

        m = self.train_metrics.compute()
        return {
            'loss': total_loss / n_batches,
            'accuracy': m['accuracy'],
            'PER': m['PER']
        }

    # ----------------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0
        n_batches = 0

        apply_fusion = (epoch >= self.config.apply_fusion_after_epoch)

        for batch in tqdm(self.val_loader, desc="Validation"):
            x = batch['waveform'].to(DEVICE)
            y = batch['phoneme'].to(DEVICE)

            logits = self.model.forward_eval(
                x,
                apply_fusion=apply_fusion
            )

            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            self.val_metrics.update(preds, y)

        m = self.val_metrics.compute()
        return {
            'loss': total_loss / n_batches,
            'accuracy': m['accuracy'],
            'PER': m['PER']
        }

    # ----------------------------------------------------------
    # CHECKPOINT
    # ----------------------------------------------------------
    def save_checkpoint(self, epoch, metrics, is_best=False):

        ckpt = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'history': self.history
        }

        if not self.config.save_best_only:
            if epoch % self.config.save_every == 0:
                torch.save(ckpt,
                           f"{self.config.save_dir}/checkpoint_epoch{epoch}.pth")

        if is_best:
            torch.save(ckpt, f"{self.config.save_dir}/best_model.pth")
            print(f"🏆 NEW BEST PER: {metrics['PER']:.2f}%")

    # ----------------------------------------------------------
    # FULL TRAINING LOOP
    # ----------------------------------------------------------
    def train(self):

        print("\n===== TRAINING START =====\n")

        start = datetime.now()

        for epoch in range(1, self.config.epochs + 1):

            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            lr = self.scheduler.step()

            self.history['train_loss'].append(train_m['loss'])
            self.history['train_per'].append(train_m['PER'])
            self.history['val_loss'].append(val_m['loss'])
            self.history['val_per'].append(val_m['PER'])
            self.history['lr'].append(lr)

            print(f"\n===== EPOCH {epoch} =====")
            print(f"Train: loss={train_m['loss']:.4f} | PER={train_m['PER']:.2f}%")
            print(f"Val:   loss={val_m['loss']:.4f} | PER={val_m['PER']:.2f}%")
            print(f"LR = {lr:.6f}")

            is_best = val_m['PER'] < self.best_per
            if is_best:
                self.best_per = val_m['PER']
                self.best_epoch = epoch

            self.save_checkpoint(epoch, val_m, is_best)

        print("\nTRAINING COMPLETE")
        print(f"Best PER = {self.best_per:.2f}% at epoch {self.best_epoch}")
        print(f"Total time = {datetime.now() - start}")

        with open(f"{self.config.save_dir}/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


# ============================================================
# MAIN
# ============================================================
def main():
    config = OptimalConfig()

    print("\n============================")
    print("TRAINING CONFIGURATION")
    print("============================")
    print(config)
    print("============================\n")

    print("Loading TIMIT dataset...")


    from dataset import TIMITDataset, pad_collate_capsnet
    train_dataset = TIMITDataset(root=config.timit_root, split='train')
    val_dataset   = TIMITDataset(root=config.timit_root, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               shuffle=True, num_workers=config.num_workers,
                               pin_memory=config.pin_memory,
                               collate_fn=pad_collate_capsnet)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=config.num_workers,
                             pin_memory=config.pin_memory,
                             collate_fn=pad_collate_capsnet)

    print("⚠ WARNING: dummy dataset used.")
    dummy = [{'waveform': torch.randn(1, 16000), 'phoneme': torch.randint(0, 61, (1,))}] * 10
    train_loader = DataLoader(dummy, batch_size=2)
    val_loader = DataLoader(dummy, batch_size=2)

    class_counts = [1000] * 61

    trainer = OptimalTrainer(config, train_loader, val_loader, class_counts)
    trainer.train()


if __name__ == "__main__":
    main()
