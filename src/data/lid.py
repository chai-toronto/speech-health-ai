import torchaudio
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-256"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

def verify_english(waveform):
    inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    lang_id = torch.argmax(outputs, dim=-1)[0].item()
    return model.config.id2label[lang_id] == 'eng'


