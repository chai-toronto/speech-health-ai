import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch

model_id = "facebook/mms-lid-256"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

def infer_lid_distribution(waveform_chunk, sample_rate, threshold=1):
    """
    Returns the language probability distribution for a waveform chunk.

    Args:
        waveform_chunk (Tensor): 1D tensor of audio samples.
        sample_rate (int): Sampling rate of the audio.
        threshold (float): top languages that explain certain percentages of the distribution. Default as 1 to return all languages.
    Returns:
        Dict[str, float]: Mapping from language code to probability.
    """
    inputs = processor(waveform_chunk, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    # Map probabilities to language labels
    lang_distribution = {
        model.config.id2label[idx]: prob.item()
        for idx, prob in enumerate(probs)
    }

    sorted_langs = sorted(lang_distribution.items(), key=lambda x: x[1], reverse=True)
    cumulative_prob = 0.0
    selected_langs = {}

    for lang, prob in sorted_langs:
        selected_langs[lang] = prob
        cumulative_prob += prob
        if cumulative_prob >= threshold:
            break

    return selected_langs


def trim_non_target_language(
    waveform,
    sample_rate,
    target_lang='eng',
    window_sec=3.0,
    hop_sec=1.0,
    chunk_threshold=0.7
):
    assert waveform.ndim == 1, f"Expected 1D Tensor, got a Tensor of shape: {waveform.shape}"

    num_samples = waveform.shape[0]

    window_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)

    retained_segments = []

    for start_sample in range(0, num_samples - window_samples + 1, hop_samples):
        chunk = waveform[start_sample: start_sample + window_samples]
        target_prob = infer_lid_distribution(chunk, sample_rate)[target_lang]

        if target_prob >= chunk_threshold:
            start_time = start_sample / sample_rate
            end_time = (start_sample + window_samples) / sample_rate
            retained_segments.append((start_time, end_time))

    # Merge adjacent or overlapping segments
    merged_segments = []
    for seg in retained_segments:
        if not merged_segments or seg[0] > merged_segments[-1][1]:
            merged_segments.append(seg)
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], seg[1]))

    # Concatenate retained audio
    trimmed_audio = []
    for start_time, end_time in merged_segments:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        trimmed_audio.append(waveform[start_sample:end_sample])

    if trimmed_audio:
        trimmed_audio = torch.cat(trimmed_audio)
    else:
        trimmed_audio = torch.empty(0)

    return trimmed_audio, merged_segments


def drop_one_segment_maximize_target_prob(
    waveform,
    sample_rate,
    target_lang='eng',
    window_sec=2.0,
    hop_sec=1.0
):
    assert waveform.ndim == 1, f"Expected 1D Tensor, got: {waveform.shape}"

    num_samples = waveform.shape[0]
    window_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)

    # Baseline probability with full audio
    baseline_prob = infer_lid_distribution(waveform, sample_rate)[target_lang]

    best_prob = baseline_prob
    best_waveform = waveform
    best_dropped = None

    for start_sample in range(0, num_samples - window_samples + 1, hop_samples):
        end_sample = start_sample + window_samples

        # Remove the segment
        modified_waveform = torch.cat([
            waveform[:start_sample],
            waveform[end_sample:]
        ])

        if len(modified_waveform) == 0:
            continue

        prob = infer_lid_distribution(modified_waveform, sample_rate)[target_lang]

        if prob > best_prob:
            best_prob = prob
            best_waveform = modified_waveform
            best_dropped = (
                start_sample / sample_rate,
                end_sample / sample_rate
            )

    return best_waveform, best_dropped, best_prob


