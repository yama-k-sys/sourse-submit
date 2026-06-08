import os
import argparse
import sounddevice as sd
import soundfile as sf
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def next_filename(folder, label):
    # 例: human_001.wav の連番を作る
    files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
    nums = []
    for f in files:
        name = f.replace(".wav", "")
        if "_" in name:
            try:
                nums.append(int(name.split("_")[-1]))
            except:
                pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(folder, f"{label}_{next_num:03d}.wav")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["human","ai"], required=True,
                        help="保存先ラベル: human or ai")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    root = "dataset"
    folder = os.path.join(root, args.label)
    ensure_dir(folder)

    print(f"\n=== 録音を開始します ===")
    print(f"ラベル: {args.label}  |  時間: {args.seconds}秒  |  サンプリング: {args.sr}Hz")
    print("3秒後に録音開始...")
    sd.sleep(1000)
    print("2...")
    sd.sleep(1000)
    print("1...")
    sd.sleep(1000)
    print("🎙 録音中...")

    audio = sd.rec(int(args.seconds * args.sr), samplerate=args.sr, channels=1)
    sd.wait()

    save_path = next_filename(folder, args.label)
    sf.write(save_path, audio, args.sr)

    print(f"✅ 保存しました → {save_path}\n")

if __name__ == "__main__":
    main()
