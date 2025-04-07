#pip install speechbrain
#pip install torch torchvision torchaudio

#preparing of dataset as csv files


#1. loading and preprocessing data
import os
import pandas as pd
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataset import DynamicItemDataset

# Load your dataset
data_csv = "path_to_your_csv_file.csv"
data = pd.read_csv(data_csv)

# Define a function to read audio files
def load_audio(data):
    path = data['path']
    audio = read_audio(path)
    return audio

# Create a dataset
dataset = DynamicItemDataset.from_csv(
    csv_path=data_csv,
    replacements={"data_root": "path_to_your_audio_files"},
    dynamic_items=[("audio", load_audio)],
    output_keys=["id", "audio", "label"]
)

#2.data pipeline
from speechbrain.dataio.dataio import read_audio

# Define a function to read audio files
def load_audio(data):
    path = data['path']
    audio = read_audio(path)
    return audio

# Create a dataset
dataset = DynamicItemDataset.from_csv(
    csv_path=data_csv,
    replacements={"data_root": "path_to_your_audio_files"},
    dynamic_items=[("audio", load_audio)],
    output_keys=["id", "audio", "label"]
)

#3.defining the model

from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from torch import nn

class EmotionClassifier(nn.Module):
    def __init__(self, wav2vec2_model):
        super(EmotionClassifier, self).__init__()
        self.wav2vec2 = wav2vec2_model
        self.classifier = nn.Linear(768, 7)  # Assuming 7 emotions

    def forward(self, audio):
        features = self.wav2vec2(audio)
        return self.classifier(features)
    

#4.training the model
from speechbrain.core import Brain
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class EmotionTraining(Brain):
    def compute_forward(self, batch, stage):
        audio = batch.audio
        outputs = self.modules.wav2vec2(audio)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label
        loss = self.hparams.criterion(predictions, labels)
        return loss

# Initialize the model and optimizer
hparams = {
    "wav2vec2_model": "facebook/wav2vec2-large-960h",
    "criterion": CrossEntropyLoss(),
    "optimizer": Adam,
    "lr": 1e-5,
}

brain = EmotionTraining(
    modules={"wav2vec2": HuggingFaceWav2Vec2(hparams["wav2vec2_model"])},
    hparams=hparams,
    run_opts={"device": "cuda"},
    opt_class=hparams["optimizer"],
    loss_fn=hparams["criterion"],
)

# Train the model
brain.fit(
    epoch_count=10,
    train_set=dataset,
    valid_set=dataset,
    train_loader_kwargs={"batch_size": 16},
    valid_loader_kwargs={"batch_size": 16},
)


#5. testing the model
brain.evaluate(test_set=dataset, test_loader_kwargs={"batch_size": 16})#pip install speechbrain
#pip install torch torchvision torchaudio

#preparing of dataset as csv files


#1. loading and preprocessing data
import os
import pandas as pd
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataset import DynamicItemDataset

# Load your dataset
data_csv = "path_to_your_csv_file.csv"
data = pd.read_csv(data_csv)

# Define a function to read audio files
def load_audio(data):
    path = data['path']
    audio = read_audio(path)
    return audio

# Create a dataset
dataset = DynamicItemDataset.from_csv(
    csv_path=data_csv,
    replacements={"data_root": "path_to_your_audio_files"},
    dynamic_items=[("audio", load_audio)],
    output_keys=["id", "audio", "label"]
)

#2.data pipeline
from speechbrain.dataio.dataio import read_audio

# Define a function to read audio files
def load_audio(data):
    path = data['path']
    audio = read_audio(path)
    return audio

# Create a dataset
dataset = DynamicItemDataset.from_csv(
    csv_path=data_csv,
    replacements={"data_root": "path_to_your_audio_files"},
    dynamic_items=[("audio", load_audio)],
    output_keys=["id", "audio", "label"]
)

#3.defining the model

from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from torch import nn

class EmotionClassifier(nn.Module):
    def __init__(self, wav2vec2_model):
        super(EmotionClassifier, self).__init__()
        self.wav2vec2 = wav2vec2_model
        self.classifier = nn.Linear(768, 7)  # Assuming 7 emotions

    def forward(self, audio):
        features = self.wav2vec2(audio)
        return self.classifier(features)
    

#4.training the model
from speechbrain.core import Brain
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class EmotionTraining(Brain):
    def compute_forward(self, batch, stage):
        audio = batch.audio
        outputs = self.modules.wav2vec2(audio)
        return outputs

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label
        loss = self.hparams.criterion(predictions, labels)
        return loss

# Initialize the model and optimizer
hparams = {
    "wav2vec2_model": "facebook/wav2vec2-large-960h",
    "criterion": CrossEntropyLoss(),
    "optimizer": Adam,
    "lr": 1e-5,
}

brain = EmotionTraining(
    modules={"wav2vec2": HuggingFaceWav2Vec2(hparams["wav2vec2_model"])},
    hparams=hparams,
    run_opts={"device": "cuda"},
    opt_class=hparams["optimizer"],
    loss_fn=hparams["criterion"],
)

# Train the model
brain.fit(
    epoch_count=10,
    train_set=dataset,
    valid_set=dataset,
    train_loader_kwargs={"batch_size": 16},
    valid_loader_kwargs={"batch_size": 16},
)


#5. testing the model
brain.evaluate(test_set=dataset, test_loader_kwargs={"batch_size": 16})
