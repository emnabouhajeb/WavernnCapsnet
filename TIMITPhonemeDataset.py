# TIMITPhonemeDataset.py
import os
import torch
from torch.utils.data import Dataset
from glob import glob
import torchaudio

PHONEME_LIST_61 = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'er', 'ax', 'ix',
    'axr', 'ax-h',
    'b', 'd', 'g', 'p', 't', 'k', 'dx', 'q',
    'jh', 'ch',
    's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh',
    'm', 'n', 'ng', 'em', 'en', 'eng', 'nx',
    'l', 'r', 'w', 'y', 'hh', 'hv', 'el',
    'pau', 'epi', 'h#'
]


class TIMITPhonemeDataset(Dataset):
    """
    TIMIT dataset with CONTEXT WINDOWS around each phoneme.
    Provides temporal context for better CapsNet learning.
    """

    def __init__(self, root='TIMIT', split='TRAIN', sample_rate=16000,
                 context_ms=100, min_duration_samples=40, max_duration_samples=16000):
        """
        Args:
            root: TIMIT root directory
            split: 'TRAIN' or 'TEST'
            context_ms: milliseconds of context before/after phoneme (default: 100ms)
            min_duration_samples: minimum phoneme duration
            max_duration_samples: maximum phoneme duration
        """
        self.root = root
        self.split = split.upper()
        self.sample_rate = sample_rate
        self.min_duration = min_duration_samples
        self.max_duration = max_duration_samples

        # Context window: ±100ms = ±1600 samples @ 16kHz
        self.context_samples = int(context_ms * sample_rate / 1000)

        self.items = []
        self.all_phonemes_set = set()

        # Find PHN files
        patterns = [
            os.path.join(self.root, '**', self.split, '**', '*.PHN'),
            os.path.join(self.root, '**', self.split, '**', '*.phn'),
        ]
        phn_list = []
        for p in patterns:
            phn_list.extend(glob(p, recursive=True))
        phn_list = sorted(list(set(phn_list)))

        if len(phn_list) == 0:
            raise RuntimeError(f"No PHN files found in {self.root}")

        for phn_path in phn_list:
            parts = phn_path.replace('\\', '/').split('/')
            if self.split not in [p.upper() for p in parts]:
                continue

            # Find WAV file
            base = os.path.splitext(phn_path)[0]
            wav_path = None
            for ext in ['.WAV', '.wav']:
                candidate = base + ext
                if os.path.exists(candidate):
                    wav_path = candidate
                    break

            if wav_path is None:
                continue

            # Parse phoneme annotations
            try:
                with open(phn_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts_line = line.strip().split()
                        if len(parts_line) < 3:
                            continue
                        start = int(parts_line[0])
                        end = int(parts_line[1])
                        lab = parts_line[2]
                        duration = end - start

                        if duration < self.min_duration or duration > self.max_duration:
                            continue

                        # Add context window
                        context_start = max(0, start - self.context_samples)
                        context_end = end + self.context_samples

                        self.items.append((wav_path, context_start, context_end, lab))
                        self.all_phonemes_set.add(lab)
            except Exception:
                continue

        # Build phoneme mapping
        self.phoneme_to_idx = {p: i for i, p in enumerate(PHONEME_LIST_61)}
        extras = sorted(list(self.all_phonemes_set - set(PHONEME_LIST_61)))
        for p in extras:
            self.phoneme_to_idx[p] = len(self.phoneme_to_idx)

        self.idx_to_phoneme = {i: p for p, i in self.phoneme_to_idx.items()}

        print(f"[TIMITPhonemeDataset] {self.split}")
        print(f"  Total phonemes: {len(self.items)}")
        print(f"  Context window: ±{context_ms}ms (±{self.context_samples} samples)")
        print(f"  Unique phonemes: {len(self.all_phonemes_set)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, ctx_start, ctx_end, lab = self.items[idx]

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Extract segment with context
        ctx_start = max(0, min(ctx_start, waveform.size(1) - 1))
        ctx_end = max(ctx_start + 1, min(ctx_end, waveform.size(1)))
        seg = waveform[:, ctx_start:ctx_end]

        target = self.phoneme_to_idx.get(lab, -1)

        return {
            'waveform': seg,
            'phoneme': torch.tensor(target, dtype=torch.long),
            'phoneme_str': lab,
            'wav_path': wav_path,
            'start': ctx_start
        }

    @property
    def all_phonemes(self):
        return [self.idx_to_phoneme[i] for i in range(len(self.idx_to_phoneme))]


def pad_collate_phoneme(batch):
    """Collate function for batching phonemes"""
    max_len = max([b['waveform'].size(1) for b in batch])
    waves = []
    targets = []
    wav_paths = []
    starts = []

    for b in batch:
        w = b['waveform']
        L = w.size(1)
        if L < max_len:
            pad = torch.zeros(1, max_len - L)
            w = torch.cat([w, pad], dim=1)
        waves.append(w)
        targets.append(b['phoneme'])
        wav_paths.append(b.get('wav_path', ''))
        starts.append(b.get('start', 0))

    waves = torch.stack(waves, dim=0)
    targets = torch.stack(targets, dim=0)

    return {
        'waveform': waves,
        'phoneme': targets,
        'wav_path': wav_paths,
        'start': starts
    }


# ===========================
# UTTERANCE DATASET (for reference, not used in current training)
# ===========================
class TIMITUtteranceDataset(Dataset):
    """Full utterance dataset for sequence-level training"""

    def __init__(self, root='TIMIT', split='TRAIN', sample_rate=16000,
                 max_duration_sec=5.0, min_phonemes=3):
        self.root = root
        self.split = split.upper()
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_sec * sample_rate)
        self.min_phonemes = min_phonemes

        self.utterances = []
        self.phoneme_to_idx = {p: i for i, p in enumerate(PHONEME_LIST_61)}
        self.idx_to_phoneme = {i: p for p, i in self.phoneme_to_idx.items()}

        patterns = [
            os.path.join(self.root, '**', self.split, '**', '*.PHN'),
            os.path.join(self.root, '**', self.split, '**', '*.phn'),
        ]
        phn_files = []
        for p in patterns:
            phn_files.extend(glob(p, recursive=True))
        phn_files = sorted(list(set(phn_files)))

        if len(phn_files) == 0:
            raise RuntimeError(f"No PHN files found")

        for phn_path in phn_files:
            parts = phn_path.replace('\\', '/').split('/')
            if self.split not in [p.upper() for p in parts]:
                continue

            base = os.path.splitext(phn_path)[0]
            wav_path = None
            for ext in ['.WAV', '.wav']:
                if os.path.exists(base + ext):
                    wav_path = base + ext
                    break

            if wav_path is None:
                continue

            try:
                phonemes = []
                boundaries = []
                with open(phn_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts_line = line.strip().split()
                        if len(parts_line) < 3:
                            continue
                        start = int(parts_line[0])
                        end = int(parts_line[1])
                        label = parts_line[2]

                        if label in self.phoneme_to_idx:
                            phonemes.append(self.phoneme_to_idx[label])
                            boundaries.append((start, end))

                if len(phonemes) >= self.min_phonemes:
                    self.utterances.append({
                        'wav_path': wav_path,
                        'phonemes': phonemes,
                        'boundaries': boundaries
                    })
            except Exception:
                continue

        print(f"[TIMITUtteranceDataset] {self.split}: {len(self.utterances)} utterances")

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        item = self.utterances[idx]
        waveform, sr = torchaudio.load(item['wav_path'])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.size(1) > self.max_samples:
            waveform = waveform[:, :self.max_samples]
            valid_phonemes = []
            valid_boundaries = []
            for p, (s, e) in zip(item['phonemes'], item['boundaries']):
                if e <= self.max_samples:
                    valid_phonemes.append(p)
                    valid_boundaries.append((s, e))
            phonemes = valid_phonemes
            boundaries = valid_boundaries
        else:
            phonemes = item['phonemes']
            boundaries = item['boundaries']

        return {
            'waveform': waveform,
            'phoneme_sequence': torch.tensor(phonemes, dtype=torch.long),
            'boundaries': boundaries,
            'wav_path': item['wav_path']
        }


def pad_collate_utterance(batch):
    """Collate function for utterances"""
    max_wav_len = max(b['waveform'].size(1) for b in batch)
    max_phoneme_len = max(b['phoneme_sequence'].size(0) for b in batch)

    waveforms = []
    phoneme_sequences = []
    sequence_lengths = []
    boundaries_list = []
    wav_paths = []

    for b in batch:
        wav = b['waveform']
        L = wav.size(1)
        if L < max_wav_len:
            pad = torch.zeros(1, max_wav_len - L)
            wav = torch.cat([wav, pad], dim=1)
        waveforms.append(wav)

        phonemes = b['phoneme_sequence']
        N = phonemes.size(0)
        if N < max_phoneme_len:
            pad = torch.full((max_phoneme_len - N,), -1, dtype=torch.long)
            phonemes = torch.cat([phonemes, pad], dim=0)
        phoneme_sequences.append(phonemes)

        sequence_lengths.append(N)
        boundaries_list.append(b['boundaries'])
        wav_paths.append(b['wav_path'])

    return {
        'waveform': torch.stack(waveforms, dim=0),
        'phoneme_sequence': torch.stack(phoneme_sequences, dim=0),
        'sequence_lengths': torch.tensor(sequence_lengths, dtype=torch.long),
        'boundaries': boundaries_list,
        'wav_path': wav_paths
    }