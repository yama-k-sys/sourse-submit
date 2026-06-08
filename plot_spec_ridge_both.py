# plot_spec_ridge_both.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_wav(path: str, target_sr: int | None = None):
    """WAV読み込み (soundfile -> scipy.io.wavfile). モノラル化あり."""
    y = None
    sr = None
    errs = []

    # soundfile
    try:
        import soundfile as sf
        y, sr = sf.read(path, always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
    except Exception as e:
        errs.append(f"soundfile failed: {e}")

    # scipy fallback
    if y is None:
        try:
            from scipy.io import wavfile
            sr, y = wavfile.read(path)
            if y.ndim == 2:
                y = y.mean(axis=1)
            if np.issubdtype(y.dtype, np.integer):
                maxv = np.iinfo(y.dtype).max
                y = (y.astype(np.float32) / maxv)
            else:
                y = y.astype(np.float32, copy=False)
        except Exception as e:
            errs.append(f"scipy.io.wavfile failed: {e}")

    if y is None or sr is None:
        raise RuntimeError("Failed to load wav.\n" + "\n".join(errs))

    # resample if requested
    if target_sr is not None and int(target_sr) != int(sr):
        try:
            import librosa
            y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)
        except Exception as e:
            raise RuntimeError(f"Resample requested but failed (need librosa). error={e}")

    return y, int(sr)


def compute_stft_mag(y, sr, n_fft, win_length, hop_length, window="hann", center=True):
    """STFT magnitude + axes"""
    import librosa
    S = librosa.stft(
        y,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
    )
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)
    return mag, freqs, times


def pick_2d_array_from_ssq_output(out):
    """ssq_stftの戻り値差を吸収して2D ndarrayを取り出す"""
    if isinstance(out, np.ndarray) and out.ndim == 2:
        return out
    if isinstance(out, (tuple, list)):
        for item in out[::-1]:
            if isinstance(item, np.ndarray) and item.ndim == 2:
                return item
    raise RuntimeError(f"Could not find 2D ndarray in ssq_stft output. type(out)={type(out)}")


def compute_sst_mag(y, sr, n_fft, win_length, hop_length):
    """SST(=ssq_stft) magnitude + axes (STFTと同じ定義で軸を作る)"""
    from ssqueezepy import ssq_stft
    out = ssq_stft(
        np.asarray(y, dtype=np.float64),
        fs=sr,
        n_fft=n_fft,
        win_len=win_length,
        hop_len=hop_length
    )
    ssq = pick_2d_array_from_ssq_output(out)
    mag = np.abs(ssq)

    # axes: librosaがある前提（このスクリプトはlibrosa依存）
    import librosa
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)
    return mag, freqs, times


def to_db(mag, amin=1e-12):
    """max基準でdB化（librosa無しでもOK）"""
    mag = np.maximum(mag, amin)
    ref = np.max(mag)
    ref = max(ref, amin)
    return 20.0 * np.log10(mag / ref)


def extract_ridge(mag, freqs, fmin=50.0, fmax=4000.0):
    """
    もっとも単純なリッジ:
    各時間フレームで (fmin..fmax内の) magnitude最大の周波数を選ぶ
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    mag2 = mag[mask, :]
    freqs2 = freqs[mask]
    idx = np.argmax(mag2, axis=0)
    ridge = freqs2[idx]
    return ridge


def plot_spec_with_ridge(mag_db, freqs, times, ridge, out_png: Path,
                         title: str, fmax_plot: float | None,
                         fontsize: int = 12, show: bool = False):
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(times, freqs, mag_db, shading="auto", cmap="viridis")
    # 白リッジ
    plt.plot(times, ridge, color="white", linewidth=2.5)

    plt.xlabel("Time [s]", fontsize=fontsize)
    plt.ylabel("Frequency [Hz]", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)

    if fmax_plot is not None:
        plt.ylim(0, fmax_plot)

    cbar = plt.colorbar()
    cbar.set_label("Magnitude [dB]", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", help="Input wav path")
    p.add_argument("--mode", choices=["stft", "sst"], default="stft",
                   help="stft: STFT spectrogram, sst: SST (ssq_stft) spectrogram")
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--win", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)
    p.add_argument("--fmin", type=float, default=50.0)
    p.add_argument("--fmax", type=float, default=4000.0, help="Ridge search frequency range max (Hz)")
    p.add_argument("--fmax_plot", type=float, default=8000.0, help="Plot frequency max (Hz)")
    p.add_argument("--seconds", type=float, default=None, help="Use only first N seconds")
    p.add_argument("--fontsize", type=int, default=12)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    y, sr = load_wav(str(wav_path), target_sr=args.sr)
    if args.seconds is not None:
        y = y[: int(args.seconds * sr)]

    if args.mode == "stft":
        mag, freqs, times = compute_stft_mag(y, sr, args.nfft, args.win, args.hop)
        suffix = "stft"
        title = f"STFT Spectrogram with Ridge: {wav_path.name}"
    else:
        mag, freqs, times = compute_sst_mag(y, sr, args.nfft, args.win, args.hop)
        suffix = "sst"
        title = f"SST Spectrogram with Ridge: {wav_path.name}"

    ridge = extract_ridge(mag, freqs, fmin=args.fmin, fmax=args.fmax)
    mag_db = to_db(mag)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    out_png = out_dir / f"{wav_path.stem}_{suffix}_spec_ridge.png"

    plot_spec_with_ridge(
        mag_db, freqs, times, ridge,
        out_png=out_png,
        title=title,
        fmax_plot=args.fmax_plot,
        fontsize=args.fontsize,
        show=args.show
    )


if __name__ == "__main__":
    main()
