# preprocess_phoneme.py
import torch
import torchaudio
from pathlib import Path
import os

RAW_DIR = "data/TIMIT"
OUTPUT_DIR = "data/TIMIT_PHONEME"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Macro-classes → dossiers
MACRO_CLASS_MAPPING = {
    'iy':5, 'ih':5, 'eh':5, 'ey':5, 'ae':5, 'aa':5, 'aw':5, 'ay':5,
    'ah':5, 'ao':5, 'oy':5, 'ow':5, 'uh':5, 'uw':5, 'ux':5,
    'er':5, 'ax':5, 'ix':5, 'axr':5, 'ax-h':5,
    'b':0, 'd':0, 'g':0, 'p':0, 't':0, 'k':0, 'dx':0, 'q':0,
    'bcl':0, 'dcl':0, 'gcl':0, 'pcl':0, 'tcl':0, 'kcl':0,
    's':2, 'sh':2, 'z':2, 'zh':2, 'f':2, 'th':2, 'v':2, 'dh':2,
    'm':3, 'n':3, 'ng':3, 'em':3, 'en':3, 'eng':3, 'nx':3,
    'l':4, 'r':4, 'w':4, 'y':4, 'hh':4, 'hv':4, 'el':4,
    'jh':1, 'ch':1,
    'pau':6, 'epi':6, 'h#':6, 'sil':6
}

CLASS_NAMES = {
    0: "Stops", 1: "Affricates", 2: "Fricatives", 3: "Nasals",
    4: "Semivowels", 5: "Vowels", 6: "Others"
}

def create_phoneme_segments():
    raw = Path(RAW_DIR)
    count = 0

    for wav_path in raw.rglob("*.WAV"):
        phn_path = wav_path.with_suffix('.PHN')
        if not phn_path.exists(): continue

        # Charger audio
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.mean(0)  # mono

        # Lire .PHN
        with open(phn_path) as f:
            lines = [l.strip().split() for l in f if len(l.strip().split()) == 3]

        # Créer segments
        for i, (start, end, phn) in enumerate(lines):
            start, end = int(start), int(end)
            segment = wav[start:end]
            if len(segment) < 100: continue  # trop court

            macro = MACRO_CLASS_MAPPING.get(phn, 6)
            class_dir = Path(OUTPUT_DIR) / CLASS_NAMES[macro] / phn
            class_dir.mkdir(parents=True, exist_ok=True)

            out_path = class_dir / f"{wav_path.stem}_{i:04d}.wav"
            torchaudio.save(out_path, segment.unsqueeze(0), 16000)
            count += 1

    print(f"Total phonème segments: {count}")
    for c in range(7):
        path = Path(OUTPUT_DIR) / CLASS_NAMES[c]
        total = sum(len(list(p.iterdir())) for p in path.iterdir())
        print(f"{CLASS_NAMES[c]}: {total} segments")

if __name__ == "__main__":
    create_phoneme_segments()