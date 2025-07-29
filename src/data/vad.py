import re

import torch
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.audio import Model


BETWEEN_SEGMENT = 0.15

HYPER_PARAMETERS = {
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}

class VAD:
    def __init__(self, device):
        self.model = Model.from_pretrained("pyannote/segmentation-3.0")
        self.model.to(device)
        self.speech_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.speech_pipeline.instantiate(HYPER_PARAMETERS)

        self.overlapped_pipeline = OverlappedSpeechDetection(segmentation=self.model)
        self.overlapped_pipeline.instantiate(HYPER_PARAMETERS)

    def find_speech_and_trim(self, waveform, sample_rate):
        vad = self.speech_pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return self.trim_speech(waveform, sample_rate, str(vad))

    def find_overlapped_and_trim(self, waveform, sample_rate):
        overlapped = self.overlapped_pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return self.trim_speech(waveform, sample_rate, str(overlapped), to_annotation=False)

    def find_speech(self, waveform, sample_rate):
        vad = self.speech_pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return str(vad)

    def find_overlapped(self, waveform, sample_rate):
        overlapped = self.overlapped_pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return str(overlapped)




def trim_speech(waveform, sample_rate, annotation_str, to_annotation=True):
    """If to_annotation is True, keep the annotated segments. Otherwise, remove them."""
    time_ranges = parse_annotation(annotation_str)

    T = waveform.shape[1]
    mask = torch.zeros(T, dtype=torch.bool) if to_annotation else torch.ones(T, dtype=torch.bool)

    for start_sec, end_sec in time_ranges:
        start_sample = max(0, int(start_sec * sample_rate))
        end_sample = min(T, int(end_sec * sample_rate))

        if to_annotation:
            mask[start_sample:end_sample] = True
        else:
            mask[start_sample:end_sample] = False

    trimmed_waveform = waveform[:, mask]
    return trimmed_waveform


def parse_annotation(annotation_str):
    # Extract start and end times in seconds
    pattern = r'\[\s*(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]'
    matches = re.findall(pattern, annotation_str)

    def time_to_seconds(t):
        h, m, s = t.split(':')
        return int(h)*3600 + int(m)*60 + float(s)

    time_ranges = [(time_to_seconds(start), time_to_seconds(end) + BETWEEN_SEGMENT) for start, end in matches]
    return time_ranges