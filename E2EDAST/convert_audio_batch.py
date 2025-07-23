import os
import subprocess

input_dir = "data/audio_raw"
output_dir = "data/audio"
os.makedirs(output_dir, exist_ok=True)

def convert_wav_ffmpeg(in_path, out_path):
    command = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", in_path,
        "-ar", "16000",       # sample rate
        "-ac", "1",           # mono
        "-sample_fmt", "s16", # 16-bit PCM
        out_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(".wav"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)
        convert_wav_ffmpeg(input_path, output_path)
        print(f"Converted: {fname}")
