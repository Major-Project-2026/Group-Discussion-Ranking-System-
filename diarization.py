import whisper
import datetime
import subprocess
import torch
import wave
import contextlib
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# Read the selected file
try:
    with open("selected_file.txt", "r") as f:
        path = f.read().strip()
except FileNotFoundError:
    print("No file selected. Please run upload.py first.")
    exit()

# Set device to CPU since you don't have a GPU
device = torch.device("cpu")

# Load the speaker embedding model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=device
)

# Ensure the file is in WAV format
if not path.lower().endswith('.wav'):
    subprocess.call(['ffmpeg', '-i', path, '-ac', '1', 'audio.wav', '-y'])
    path = 'audio.wav'

# Load Whisper model
model_size = "tiny"
model = whisper.load_model(model_size , device="cpu")

# Transcribe audio
result = model.transcribe(path)
segments = result["segments"]

# Get audio duration
with contextlib.closing(wave.open(path, 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# Initialize Pyannote audio processing
audio = Audio()

def segment_embedding(segment):
    start = segment["start"]
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    return embedding_model(waveform[None])

# Compute embeddings
embeddings = np.zeros(shape=(len(segments), 192))
for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(segment)

embeddings = np.nan_to_num(embeddings)

# Perform clustering
num_speakers = 7  # Adjust based on your data
clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
labels = clustering.labels_

# Assign speaker labels to segments
for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

# Convert time to readable format
def time(secs):
    return datetime.timedelta(seconds=round(secs))

# Write transcript with speaker labels
with open("transcript.txt", "w" ,  encoding="utf-8") as f:
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
        f.write(segment["text"][1:] + ' ')

print("Processing complete! Transcript saved in transcript.txt")
