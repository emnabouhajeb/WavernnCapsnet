
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# PHONEME TAXONOMY
# ==========================================
MACRO_CLASSES = {
    'Vowels': ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao',
               'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h'],
    'Stops': ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q'],
    'Fricatives': ['s', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh', 'hv'],
    'Nasals': ['m', 'n', 'ng', 'em', 'en', 'eng', 'nx'],
    'Affricates': ['jh', 'ch'],
    'Semivowels': ['l', 'r', 'w', 'y', 'el'],
    'Others': ['pau', 'epi', 'h#']
}

ALL_PHONEMES = []
for phonemes in MACRO_CLASSES.values():
    ALL_PHONEMES.extend(phonemes)
PHONEME_TO_ID = {p: i for i, p in enumerate(sorted(set(ALL_PHONEMES)))}
ID_TO_PHONEME = {i: p for p, i in PHONEME_TO_ID.items()}

PHONEME_TO_MACRO = {}
for macro, phonemes in MACRO_CLASSES.items():
    for p in phonemes:
        PHONEME_TO_MACRO[p] = macro

# ==========================================
# PREPROCESSING
# ==========================================
def mu_law_companding(waveform: torch.Tensor, mu: int = 255) -> torch.Tensor:
    sign = torch.sign(waveform)
    abs_wave = torch.abs(waveform)
    compressed = sign * torch.log1p(mu * abs_wave) / np.log(1 + mu)
    return compressed

def voice_activity_detection(waveform: torch.Tensor,
                             threshold_db: float = -25.0,
                             frame_length: int = 400) -> torch.Tensor:
    energy = waveform ** 2
    n_frames = len(energy) // frame_length
    if n_frames == 0:
        return waveform
    energy_frames = energy[:n_frames * frame_length].reshape(n_frames, frame_length)
    frame_energy = energy_frames.mean(dim=1)
    energy_db = 10 * torch.log10(frame_energy + 1e-10)
    threshold = energy_db.max() + threshold_db
    mask = energy_db > threshold
    if mask.sum() == 0:
        return waveform
    sample_mask = mask.repeat_interleave(frame_length)
    active_samples = waveform[:len(sample_mask)][sample_mask]
    return active_samples if len(active_samples) > 0 else waveform

def preprocess_waveform(waveform: torch.Tensor,
                        apply_vad: bool = True,
                        apply_companding: bool = True,
                        target_sr: int = 16000) -> torch.Tensor:
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    if apply_vad and len(waveform) > 400:
        waveform = voice_activity_detection(waveform)
    if apply_companding:
        waveform = mu_law_companding(waveform)
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform

# ==========================================
# PHONEME ALIGNMENT PARSER
# ==========================================
def parse_phn_file(phn_path: Path) -> List[Tuple[int, int, str]]:
    segments = []
    with open(phn_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phoneme = int(parts[0]), int(parts[1]), parts[2]
                segments.append((start, end, phoneme))
    return segments


class TIMITDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 macro_class: Optional[str] = None,
                 max_samples: Optional[int] = None,
                 target_sr: int = 16000,
                 apply_preprocessing: bool = True,
                 min_duration_ms: int = 50,
                 max_duration_ms: int = 5000):
        self.root_dir = Path(root_dir)
        self.split = split
        self.macro_class = macro_class
        self.target_sr = target_sr
        self.apply_preprocessing = apply_preprocessing
        self.min_samples = int(min_duration_ms * target_sr / 1000)
        self.max_samples = int(max_duration_ms * target_sr / 1000)
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples for {split}")
        if macro_class:
            print(f"  Macro-class: {macro_class}")

    def _load_samples(self) -> List[Dict]:
        samples = []
        split_dir = self.root_dir / self.split.upper()
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        wav_files = list(split_dir.rglob("*.WAV"))
        for wav_path in wav_files:
            phn_path = wav_path.with_suffix('.PHN')
            if not phn_path.exists():
                continue
            segments = parse_phn_file(phn_path)
            for start, end, phoneme in segments:
                phoneme = phoneme.lower().strip()
                if phoneme not in PHONEME_TO_ID:
                    continue
                if self.macro_class is not None and PHONEME_TO_MACRO.get(phoneme) != self.macro_class:
                    continue
                duration = end - start
                if duration < self.min_samples or duration > self.max_samples:
                    continue
                samples.append({
                    'wav_path': wav_path,
                    'start': start,
                    'end': end,
                    'phoneme': phoneme,
                    'phoneme_id': PHONEME_TO_ID[phoneme],
                    'macro_class': PHONEME_TO_MACRO[phoneme]
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = torchaudio.load(sample['wav_path'])
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            ratio = self.target_sr / sr
            start = int(sample['start'] * ratio)
            end = int(sample['end'] * ratio)
        else:
            start = sample['start']
            end = sample['end']
        waveform = waveform[:, start:end].squeeze(0)
        if self.apply_preprocessing:
            waveform = preprocess_waveform(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return {
            'waveform': waveform,
            'phoneme': sample['phoneme_id'],
            'phoneme_str': sample['phoneme'],
            'macro_class': sample['macro_class'],
            'length': waveform.shape[-1]
        }

# ==========================================
# DATASET POUR WAVERNN
# ==========================================
class MacroClassDataset(Dataset):
    def __init__(self,
                 root_dir: str = 'TIMIT',
                 split: str = 'train',
                 macro_class: str = 'Vowels',
                 max_samples: Optional[int] = None,
                 **kwargs):
        self.base_dataset = TIMITDataset(
            root_dir=root_dir,
            split=split,
            macro_class=macro_class,
            max_samples=max_samples,
            **kwargs
        )
        self.phonemes_in_class = sorted(set(
            s['phoneme'] for s in self.base_dataset.samples
        ))
        self.phoneme_to_local_id = {p: i for i, p in enumerate(self.phonemes_in_class)}
        print(f"Phonemes in {macro_class}: {len(self.phonemes_in_class)}")
        print(f"  {self.phonemes_in_class}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        phoneme_str = item['phoneme_str']  # string
        local_id = self.phoneme_to_local_id[phoneme_str]  # map local ID
        global_id = item['phoneme']  # garder l'ID global
        return {
            'waveform': item['waveform'],
            'phoneme': local_id,  # ID local dans la macro-classe
            'phoneme_global': global_id,
            'phoneme_str': phoneme_str,
            'length': item['length']
        }


# ==========================================
# COLLATE FUNCTIONS
# ==========================================
def pad_collate_wavernn(batch):
    batch = [b for b in batch if b['waveform'].shape[-1] > 0]
    if len(batch) == 0:
        return None
    max_len = max(b['waveform'].shape[-1] for b in batch)
    waveforms, phonemes, lengths = [], [], []
    for b in batch:
        wave = b['waveform']
        pad_len = max_len - wave.shape[-1]
        if pad_len > 0:
            wave = torch.nn.functional.pad(wave, (0, pad_len))
        waveforms.append(wave)
        phonemes.append(b['phoneme'])
        lengths.append(b['length'])
    return {
        'waveform': torch.stack(waveforms),
        'phoneme': torch.LongTensor(phonemes),
        'length': torch.LongTensor(lengths)
    }

def pad_collate_capsnet(batch):
    batch = [b for b in batch if b['waveform'].shape[-1] > 0]
    if len(batch) == 0:
        return None
    max_len = max(b['waveform'].shape[-1] for b in batch)
    waveforms, phonemes = [], []
    for b in batch:
        wave = b['waveform']
        pad_len = max_len - wave.shape[-1]
        if pad_len > 0:
            wave = torch.nn.functional.pad(wave, (0, pad_len))
        waveforms.append(wave)
        phonemes.append(b['phoneme'])
    return {
        'waveform': torch.stack(waveforms),
        'phoneme': torch.LongTensor(phonemes)
    }
