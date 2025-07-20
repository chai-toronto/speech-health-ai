import re

import torch
from pyannote.audio.pipelines import VoiceActivityDetection

BETWEEN_SEGMENT = 0.15

# instantiate the model
from pyannote.audio import Model
model = Model.from_pretrained(
  "pyannote/segmentation-3.0")
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)

def vad_and_trim(waveform, sample_rate):
    vad = pipeline({'waveform': waveform, 'sample_rate': sample_rate})
    return trim_silence(waveform, sample_rate, str(vad))

def parse_annotation(annotation_str):
    # Extract start and end times in seconds
    pattern = r'\[\s*(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]'
    matches = re.findall(pattern, annotation_str)

    def time_to_seconds(t):
        h, m, s = t.split(':')
        return int(h)*3600 + int(m)*60 + float(s)

    time_ranges = [(time_to_seconds(start), time_to_seconds(end) + BETWEEN_SEGMENT) for start, end in matches]
    return time_ranges


def trim_silence(waveform, sample_rate, annotation_str):
    time_ranges = parse_annotation(annotation_str)
    speech_segments = []

    for start_sec, end_sec in time_ranges:
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        speech_segments.append(waveform[:, start_sample:end_sample])

    if speech_segments:
        trimmed_waveform = torch.cat(speech_segments, dim=1)
    else:
        trimmed_waveform = torch.zeros((waveform.shape[0], 0))

    return trimmed_waveform