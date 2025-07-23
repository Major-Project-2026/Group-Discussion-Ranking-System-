# import whisper
# import librosa
# import torch
# import numpy as np
# import os
# import json
# import torchaudio
# from transformers import pipeline, Wav2Vec2FeatureExtractor, HubertForSequenceClassification

# # Load Whisper ASR model
# asr_model = whisper.load_model("base")

# # Load text-based emotion classification model
# text_emotion_model = pipeline(
#     "text-classification",
#     model="j-hartmann/emotion-english-distilroberta-base",
#     top_k=1  # Use top_k instead of deprecated return_all_scores
# )

# # Load audio-based emotion model and feature extractor
# processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
# audio_model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")

# def audio_emotion_detection(audio_chunk, sr):
#     # Convert numpy to torch tensor
#     if isinstance(audio_chunk, np.ndarray):
#         audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
#     else:
#         audio_tensor = audio_chunk

#     # Convert stereo to mono
#     if audio_tensor.ndim > 1:
#         audio_tensor = audio_tensor.mean(dim=0)

#     # Resample if needed
#     if sr != 16000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#         audio_tensor = resampler(audio_tensor)

#     # Ensure it's 1D
#     if audio_tensor.ndim > 1:
#         audio_tensor = audio_tensor.mean(dim=0)

#     # Remove batch dim — processor will add it
#     inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         logits = audio_model(**inputs).logits

#     probs = torch.softmax(logits, dim=1)[0]
#     labels = audio_model.config.id2label
#     top_score, top_idx = torch.max(probs, dim=0)

#     return {
#         "label": labels[top_idx.item()],
#         "score": round(top_score.item(), 3),
#         "full_probs": {labels[i]: round(probs[i].item(), 3) for i in range(len(probs))}
#     }

# def analyze_audio(audio_path):
#     print("🔍 Transcribing with Whisper...")
#     result = asr_model.transcribe(audio_path, word_timestamps=True)
#     segments = result["segments"]

#     print("🎧 Loading audio file...")
#     audio, sr = librosa.load(audio_path, sr=16000)

#     print("📊 Processing segments...")
#     final_output = []

#     for segment in segments:
#         start, end, text = segment["start"], segment["end"], segment["text"].strip()
#         start_sample = int(start * sr)
#         end_sample = int(end * sr)
#         chunk = audio[start_sample:end_sample]

#         if len(chunk) < 1000:
#             continue  # Skip very short audio segments

#         # Text-based emotion
#         text_emotions = text_emotion_model(text)
#         top_text_emotion = text_emotions[0][0]

#         # Audio-based emotion
#         audio_emotion = audio_emotion_detection(chunk, sr)

#         final_output.append({
#             "start": round(start, 2),
#             "end": round(end, 2),
#             "text": text,
#             "text_emotion": {
#                 "label": top_text_emotion["label"],
#                 "score": round(top_text_emotion["score"], 3)
#             },
#             "audio_emotion": audio_emotion
#         })

#     return final_output

# def save_results(data, output_file="output/analysis_output.json"):
#     os.makedirs("output", exist_ok=True)
#     with open(output_file, "w") as f:
#         json.dump(data, f, indent=2)
#     print(f"✅ Results saved to {output_file}")

# if __name__ == "__main__":
#     audio_file = "audio/output.wav"  # 🔁 Replace with your actual file path
#     results = analyze_audio(audio_file)
#     save_results(results)


#########################################
# import whisper
# import librosa
# import torch
# import numpy as np
# import os
# import json
# import torchaudio
# from transformers import pipeline, Wav2Vec2FeatureExtractor, HubertForSequenceClassification
# from pyannote.audio import Pipeline

# # 🔐 HuggingFace Token for gated model
# HUGGINGFACE_TOKEN = "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"

# # 🧠 Load Whisper ASR model
# asr_model = whisper.load_model("base")

# # 🗣️ Text-based emotion model
# text_emotion_model = pipeline(
#     "text-classification",
#     model="j-hartmann/emotion-english-distilroberta-base",
#     top_k=1
# )

# # 🔊 Audio-based emotion model
# processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
# audio_model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")

# # 🔈 Load speaker diarization pipeline
# print("🔈 Loading speaker diarization model...")
# diarization_pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization",
#     use_auth_token=HUGGINGFACE_TOKEN
# )

# # 🧪 Detect audio-based emotion
# def audio_emotion_detection(audio_chunk, sr):
#     if isinstance(audio_chunk, np.ndarray):
#         audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
#     else:
#         audio_tensor = audio_chunk

#     if audio_tensor.ndim > 1:
#         audio_tensor = audio_tensor.mean(dim=0)

#     if sr != 16000:
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#         audio_tensor = resampler(audio_tensor)

#     if audio_tensor.ndim > 1:
#         audio_tensor = audio_tensor.mean(dim=0)

#     inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         logits = audio_model(**inputs).logits

#     probs = torch.softmax(logits, dim=1)[0]
#     labels = audio_model.config.id2label
#     top_score, top_idx = torch.max(probs, dim=0)

#     return {
#         "label": labels[top_idx.item()],
#         "score": round(top_score.item(), 3),
#         "full_probs": {labels[i]: round(probs[i].item(), 3) for i in range(len(probs))}
#     }

# # 🎙️ Get speaker segments
# def get_speaker_segments(audio_path, num_speakers=None):
#     if num_speakers:
#         diarization = diarization_pipeline(audio_path, num_speakers=int(num_speakers))
#     else:
#         diarization = diarization_pipeline(audio_path)

#     speaker_map = {}
#     speaker_idx = 1
#     speaker_segments = []

#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         if speaker not in speaker_map:
#             speaker_map[speaker] = f"Speaker {speaker_idx}"
#             speaker_idx += 1
#         speaker_segments.append({
#             "start": round(turn.start, 2),
#             "end": round(turn.end, 2),
#             "speaker": speaker_map[speaker]
#         })

#     return speaker_segments

# # 👤 Match each segment to speaker
# def find_speaker_for_segment(start, end, speaker_segments):
#     for seg in speaker_segments:
#         if abs(seg["start"] - start) < 0.5 and abs(seg["end"] - end) < 0.5:
#             return seg["speaker"]
#         if seg["start"] <= start < seg["end"] or seg["start"] < end <= seg["end"]:
#             return seg["speaker"]
#     return "Unknown"

# # 🔍 Full analysis
# def analyze_audio(audio_path, num_speakers=None):
#     print("🔍 Transcribing with Whisper...")
#     result = asr_model.transcribe(audio_path, word_timestamps=True)
#     segments = result["segments"]

#     print("🔎 Performing speaker diarization...")
#     speaker_segments = get_speaker_segments(audio_path, num_speakers)

#     print("🎧 Loading audio file...")
#     audio, sr = librosa.load(audio_path, sr=16000)

#     print("📊 Processing segments...")
#     final_output = []

#     for segment in segments:
#         start, end, text = segment["start"], segment["end"], segment["text"].strip()
#         start_sample = int(start * sr)
#         end_sample = int(end * sr)
#         chunk = audio[start_sample:end_sample]

#         if len(chunk) < 1000:
#             continue

#         text_emotions = text_emotion_model(text)
#         top_text_emotion = text_emotions[0][0]

#         audio_emotion = audio_emotion_detection(chunk, sr)

#         speaker = find_speaker_for_segment(start, end, speaker_segments)

#         final_output.append({
#             "start": round(start, 2),
#             "end": round(end, 2),
#             "speaker": speaker,
#             "text": text,
#             "text_emotion": {
#                 "label": top_text_emotion["label"],
#                 "score": round(top_text_emotion["score"], 3)
#             },
#             "audio_emotion": audio_emotion
#         })

#     return final_output

# # 💾 Save results
# def save_results(data, output_file="output/analysis_output.json"):
#     os.makedirs("output", exist_ok=True)
#     with open(output_file, "w") as f:
#         json.dump(data, f, indent=2)
#     print(f"✅ Results saved to {output_file}")

# # 🏁 Entry point
# if __name__ == "__main__":
#     audio_file = "audio/output.wav"  # Replace as needed

#     # Ask user for optional speaker count
#     try:
#         speaker_input = input("How many speakers are in the audio? (Leave blank to auto-detect): ").strip()
#         num_speakers = int(speaker_input) if speaker_input else None
#     except ValueError:
#         num_speakers = None

#     results = analyze_audio(audio_file, num_speakers)
#     save_results(results)






import os
import torch
import whisper
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from pydub import AudioSegment
import tempfile
import torchaudio

# Load models
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔧 Device: {device}")

HUGGINGFACE_TOKEN = "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"

print("🔈 Loading speaker diarization model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)

print("🧠 Loading Whisper for transcription...")
asr_model = whisper.load_model("base")

print("😊 Loading text emotion classifier...")
text_classifier = hf_pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

print("🎧 Loading audio emotion model...")
audio_emotion_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
audio_emotion_model.eval()

# Simple function to classify text emotion
def get_text_emotion(text):
    result = text_classifier(text)[0]
    return {"label": result["label"], "score": round(result["score"], 3)}

# Audio emotion stub (replace with your own model if needed)
def get_audio_emotion(start, end, audio_path):
    segment = AudioSegment.from_file(audio_path)
    subsegment = segment[int(start * 1000):int(end * 1000)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        subsegment.export(tmp.name, format="wav")
        wav, sr = torchaudio.load(tmp.name)
        os.remove(tmp.name)

    with torch.no_grad():
        features = audio_emotion_model.forward(wav.mean(dim=0).unsqueeze(0))
        probs = torch.softmax(features, dim=1)[0]

    emotion_labels = ["neu", "hap", "ang", "sad"]
    top_idx = torch.argmax(probs).item()

    return {
        "label": emotion_labels[top_idx],
        "score": round(probs[top_idx].item(), 3),
        "full_probs": {label: round(probs[i].item(), 3) for i, label in enumerate(emotion_labels)}
    }

# Diarization + emotion merging function
def combine_diarization_with_emotions(diarization, segments, audio_path):
    diarization_segments = list(diarization.itertracks(yield_label=True))
    diarization_segments.sort(key=lambda x: x[0].start)

    output = []
    current_speaker = None
    current_entry = None

    for turn, _, speaker in diarization_segments:
        start = turn.start
        end = turn.end

        # Match transcript text within this time window
        text_parts = [seg['text'] for seg in segments if seg['start'] >= start and seg['end'] <= end]
        text = " ".join(text_parts).strip()

        if not text:
            continue

        if current_speaker != speaker:
            if current_entry:
                output.append(current_entry)
            current_speaker = speaker

            text_emotion = get_text_emotion(text)
            audio_emotion = get_audio_emotion(start, end, audio_path)

            current_entry = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
                "text_emotion": text_emotion,
                "audio_emotion": audio_emotion
            }
        else:
            current_entry['end'] = end
            current_entry['text'] += " " + text

    if current_entry:
        current_entry["text_emotion"] = get_text_emotion(current_entry["text"])
        current_entry["audio_emotion"] = get_audio_emotion(current_entry["start"], current_entry["end"], audio_path)
        output.append(current_entry)

    return output

# Transcribe function
def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path, word_timestamps=True)
    segments = result["segments"]
    return segments

# Main function
def main():
    audio_path = "audio/output.wav"  # Replace with your file
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return

    print("🔍 Performing diarization...")
    diarization = diarization_pipeline(audio_path)

    print("🗣️ Transcribing...")
    segments = transcribe_audio(audio_path)

    print("🔗 Combining results...")
    final_output = combine_diarization_with_emotions(diarization, segments, audio_path)

    for entry in final_output:
        print(entry)

if __name__ == "__main__":
    main()

