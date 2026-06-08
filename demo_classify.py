#!/usr/bin/env python3
# demo_classify.py (fixed ssq_stft dtype fallback + robust features & training)
import os, argparse, pickle, warnings
import numpy as np
import sounddevice as sd
import soundfile as sf
from tqdm import tqdm

# optional libs
try:
    import librosa
except Exception:
    librosa = None
try:
    from ssqueezepy import ssq_stft
except Exception:
    raise

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset"
MODEL_PATH  = "model.pkl"
SR = 16000

# ----------------- robust FSST call -----------------
def safe_ssq_stft(y, sr, n_fft, win_len, hop_len):
    """
    Try to call ssq_stft with dtype=np.float64 first; if that fails due to dtype handling,
    call without dtype. Return (Wx, ssq) as ssqueezepy returns.
    """
    y64 = np.asarray(y, dtype=np.float64).flatten()
    try:
        # Try with dtype (may fail in some ssqueezepy builds/environments)
        Wx, ssq, *rest = ssq_stft(y64, fs=sr, n_fft=n_fft, win_len=win_len, hop_len=hop_len, dtype=np.float64, padtype='reflect')
        return Wx, ssq
    except TypeError as e:
        # Fallback: call without dtype argument
        warnings.warn("ssq_stft dtype call failed, falling back to call without dtype: " + str(e))
        Wx, ssq, *rest = ssq_stft(y64, fs=sr, n_fft=n_fft, win_len=win_len, hop_len=hop_len, padtype='reflect')
        return Wx, ssq
    except Exception as e:
        # Last resort: re-raise but with message
        raise RuntimeError("ssq_stft call failed: " + str(e))

# ----------------- feature extraction -----------------
def extract_ridge_features(y, sr=SR, n_fft=1024, win_length=960, hop_length=128):
    """
    Returns feature vector:
      [d1_mean_abs, d1_std, d2_mean_abs, d2_std, ridge_range, ridge_mean, ridge_std, energy_mean]
    """
    y = np.asarray(y, dtype=np.float64).flatten()

    # optional trim using librosa if available
    if librosa is not None:
        try:
            yt, _ = librosa.effects.trim(y, top_db=40)
            if len(yt) >= 1:
                y = yt
        except Exception:
            pass

    # normalize RMS to reduce loudness effect
    eps = 1e-12
    rms = np.sqrt(np.mean(y**2)) + eps
    if rms > 0:
        y = y / rms

    # compute SSQ-STFT safely
    Wx, ssq = safe_ssq_stft(y, sr, n_fft=n_fft, win_len=win_length, hop_len=hop_length)

    S = np.abs(ssq)  # (F, T)

    # determine freq axis robustly
    freq_axis = None
    try:
        if isinstance(Wx, (list, tuple)) and len(Wx) > 0:
            # Wx[0] often holds frequency vector
            fa = np.asarray(Wx[0], dtype=np.float64)
            if fa.shape[0] == S.shape[0]:
                freq_axis = fa
    except Exception:
        freq_axis = None

    if freq_axis is None:
        freq_axis = np.linspace(0.0, float(sr)/2.0, S.shape[0])

    # ridge: frequency (Hz) of max energy per frame
    idx = np.argmax(S, axis=0)         # indices length T
    ridge = freq_axis[idx]             # Hz values length T
    ridge = np.nan_to_num(ridge, nan=0.0, posinf=0.0, neginf=0.0)

    # diffs
    if ridge.size >= 2:
        d1 = np.diff(ridge)
    else:
        d1 = np.array([0.0])
    if d1.size >= 2:
        d2 = np.diff(d1)
    else:
        d2 = np.array([0.0])

    feats = [
        float(np.mean(np.abs(d1))) if d1.size>0 else 0.0,
        float(np.std(d1, ddof=1)) if d1.size>1 else 0.0,
        float(np.mean(np.abs(d2))) if d2.size>0 else 0.0,
        float(np.std(d2, ddof=1)) if d2.size>1 else 0.0,
        float(np.max(ridge)-np.min(ridge)) if ridge.size>0 else 0.0,
        float(np.mean(ridge)) if ridge.size>0 else 0.0,
        float(np.std(ridge, ddof=1)) if ridge.size>1 else 0.0,
        float(np.mean(S)) if S.size>0 else 0.0
    ]
    return feats

# ----------------- build dataset features -----------------
def build_dataset_features(dataset_dir=DATASET_DIR, sr=SR):
    X = []
    y = []
    paths = []
    for label_name, label_val in [("human",1), ("ai",0)]:
        folder = os.path.join(dataset_dir, label_name)
        if not os.path.isdir(folder):
            print("Warning: dataset folder missing:", folder)
            continue
        files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".wav")]
        for fn in files:
            p = os.path.join(folder, fn)
            try:
                wav, sr_file = sf.read(p)
                if sr_file != sr:
                    if librosa is not None:
                        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr_file, target_sr=sr)
                    else:
                        # if no librosa, try naive trunc/pad or skip
                        warnings.warn(f"Sample {p} has sr={sr_file} but librosa not available; attempt to continue.")
                feats = extract_ridge_features(wav, sr=sr)
                X.append(feats)
                y.append(label_val)
                paths.append(p)
            except Exception as e:
                print("Failed to process", p, ":", e)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    return X, y, paths

# ----------------- training -----------------
def train_and_save(model_out=MODEL_PATH):
    print("\n=== Training model from dataset ===")
    X, y, paths = build_dataset_features()
    print("Built X shape:", X.shape, "y shape:", y.shape)
    if X.shape[0] < 4:
        print("Not enough data to train (need >=4).")
        return False

    # sanity: show basic stats per feature
    feature_names = ["d1_mean_abs","d1_std","d2_mean_abs","d2_std","ridge_range","ridge_mean","ridge_std","energy_mean"]
    for i, name in enumerate(feature_names):
        col = X[:,i]
        print(f"{name}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    acc = clf.score(Xte, yte) if Xte.shape[0] > 0 else clf.score(Xtr, ytr)
    print(f"Validation accuracy: {acc:.3f}")

    with open(model_out, "wb") as f:
        pickle.dump({"scaler":scaler, "clf":clf, "feat_names":feature_names}, f)
    print("Saved model to", model_out)
    return True

# ----------------- record & predict -----------------
def record(seconds=5, sr=SR):
    print(f"Recording {seconds} sec (sr={sr})... Speak now.")
    audio = sd.rec(int(seconds*sr), samplerate=sr, channels=1, dtype='float64')
    sd.wait()
    return audio.flatten()

def predict_from_model(wav, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        return None
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    feats = extract_ridge_features(wav, sr=SR)
    X = np.array([feats], dtype=float)
    Xs = obj["scaler"].transform(X)
    proba = obj["clf"].predict_proba(Xs)[0]
    # classes are [0,1] -> assume 1 is human
    if hasattr(obj["clf"], "classes_") and 1 in obj["clf"].classes_:
        idx = list(obj["clf"].classes_).index(1)
    else:
        idx = 1 if len(proba)>1 else 0
    human_pct = float(proba[idx]*100.0)
    return human_pct

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotrain", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    if args.autotrain:
        ok = train_and_save()
        if not ok:
            print("Training failed or insufficient data.")
        return

    # record and predict demo
    wav = record(args.seconds, SR)
    sf.write("recorded.wav", wav, SR)
    human_pct = predict_from_model(wav)
    if human_pct is None:
        print("No trained model present. Run with --autotrain to train from dataset/")
    else:
        print(f"判定: あなたは {human_pct:.1f}% 人間です")

if __name__ == "__main__":
    main()
