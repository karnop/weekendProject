import numpy as np
import librosa
import json
from SpeechAnalysis.LLMReasoning import reasoning


def downsample_series(times, values, target_sr=1.0):
    """Downsample time series to ~target_sr Hz (default = 1 value/sec)."""
    if len(times) == 0 or len(values) == 0:
        return []

    duration = times[-1]
    step = 1.0 / target_sr
    bins = np.arange(0, duration + step, step)

    ds_points = []
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (times >= start) & (times < end)
        if np.any(mask):
            avg_val = float(np.mean(values[mask]))
            ds_points.append({"time": float((start + end) / 2), "value": avg_val})

    return ds_points


def normalize(value, min_value, max_value, invert=False):
    """Normalize features to [0, 1]."""
    v = max(min_value, min(value, max_value))
    norm = (v - min_value) / (max_value - min_value) if max_value > min_value else 0
    return 1 - norm if invert else norm


import numpy as np
import librosa


def speechAnalysis(data, y, sr):
    duration = float(librosa.get_duration(y=y, sr=sr))

    # extracting text features
    words = data["results"]["channels"][0]["alternatives"][0]["words"]

    # inter-word pauses
    pauses, pause_timeline = [], []
    for i in range(1, len(words)):
        gap = float(words[i]["start"] - words[i - 1]["end"])
        if gap > 0:
            pauses.append(gap)
            pause_timeline.append({"time": float(words[i - 1]["end"]), "pause": gap})

    avg_pause = float(np.mean(pauses)) if pauses else 0.0
    long_pause_ratio = float(np.mean([p > 0.5 for p in pauses])) if pauses else 0.0

    # speech rate (WPM)
    total_words = len(words)
    speech_rate = float(total_words / duration) if duration > 0 else 0.0
    wpm = speech_rate * 60.0

    # speech rate stability per sentence
    sentences, speech_rate_timeline = [], []
    for p in data["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]:
        for s in p["sentences"]:
            sent_duration = float(s["end"] - s["start"])
            sent_word_count = len(s["text"].split())
            if sent_duration > 0:
                rate = sent_word_count / sent_duration
                sentences.append(rate)
                speech_rate_timeline.append(
                    {"start": float(s["start"]), "end": float(s["end"]),
                     "wps": float(rate), "wpm": float(rate * 60)}
                )

    speech_rate_std = float(np.std(sentences)) if sentences else 0.0
    speech_rate_stability = 1.0 / (1.0 + speech_rate_std)

    # filler density
    fillers = {"um", "uh", "like", "you know"}
    filler_count = int(sum(1 for w in words if w["word"].lower() in fillers))
    filler_density = filler_count / total_words if total_words > 0 else 0.0
    filler_occurrences = [{"time": float(w["start"]), "word": w["word"]}
                          for w in words if w["word"].lower() in fillers]

    # pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0

    # energy (RMS loudness) + frame times
    frame_length, hop_length = 2048, 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # raw pitch & energy series
    n_frames = pitches.shape[1]
    raw_pitch = np.array([np.mean(pitches[:, i]) for i in range(n_frames)], dtype=float)
    raw_energy = rms.astype(float)

    # downsampled to 1 point/sec
    pitch_contour = downsample_series(times, raw_pitch, target_sr=1.0)
    energy_timeline = downsample_series(times, raw_energy, target_sr=1.0)

    energy_std = float(np.std(rms)) if len(rms) > 0 else 0.0

    # voicing ratio
    voice_frames = int(np.sum(rms > (0.1 * np.max(rms))))
    voicing_ratio = float(voice_frames / len(rms)) if len(rms) > 0 else 0.0

    # smoothness
    pitch_derivative = np.diff(pitch_values) if len(pitch_values) > 1 else [0]
    pitch_smoothness = float(1.0 / (1.0 + np.var(pitch_derivative)))

    energy_derivative = np.diff(rms) if len(rms) > 1 else [0]
    energy_smoothness = float(1.0 / (1.0 + np.var(energy_derivative)))

    # normalization
    APD_norm = float(normalize(avg_pause, 0, 1, invert=True))
    LPR_norm = float(normalize(long_pause_ratio, 0, 1, invert=True))
    SR_norm = float(normalize(speech_rate, 1, 4))
    SRS_norm = float(speech_rate_stability)
    FD_norm = float(normalize(filler_density, 0, 0.1, invert=True))

    PV_norm = float(normalize(pitch_std, 0, 100, invert=True))
    EV_norm = float(normalize(energy_std, 0, 0.1, invert=True))
    VR_norm = float(normalize(voicing_ratio, 0.5, 1))
    PS_norm = float(pitch_smoothness)
    ES_norm = float(energy_smoothness)

    fluency_score = (
        0.1 * APD_norm + 0.1 * LPR_norm + 0.15 * SR_norm + 0.1 * SRS_norm +
        0.05 * FD_norm + 0.1 * PV_norm + 0.1 * EV_norm +
        0.1 * VR_norm + 0.1 * PS_norm + 0.1 * ES_norm
    )

    # ✅ Pure Python dict, JSON-compatible
    fluencyJson = {
        "Fluency Score (0-1)": round(float(fluency_score), 3),
        "Fluency Score (%)": f"{fluency_score*100:.1f}%",
        "WPM": float(wpm),
        "Average Pause Duration": float(avg_pause),
        "Filler Words": int(filler_count),
        "Energy / Loudness": float(energy_std),
        "Pitch Variation (Prosody)": float(pitch_std),
    }

    graphs = {
        "PauseTimeline": pause_timeline,
        "SpeechRateTimeline": speech_rate_timeline,
        "FillerOccurrences": filler_occurrences,
        "PitchContour": pitch_contour,
        "EnergyTimeline": energy_timeline,
    }

    llmreason = reasoning(
        average_pause_sec=avg_pause, 
        long_pause_ratio=long_pause_ratio, 
        speech_rate_wps=speech_rate, 
        speech_rate_stability= speech_rate_stability,
        filler_density=filler_density, 
        pitch_variability=pitch_std, 
        energy_variability=energy_std, 
        voicing_ratio=voicing_ratio, 
        pitch_smoothness=pitch_smoothness, 
        energy_smoothness=energy_smoothness, 
        composite_fluency_score=fluency_score
    )


    combined_dict = {}
    combined_dict.update(fluencyJson)
    combined_dict.update(llmreason)
    combined_dict.update(graphs)


    return combined_dict
