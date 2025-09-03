import numpy as np
import librosa
from SpeechAnalysis.LLMReasoning import reasoning

# normalizing features
def normalize(value, min_value, max_value, invert=False):
    v = max(min_value, min(value, max_value))
    norm = (v - min_value) / (max_value - min_value) if max_value > min_value else 0
    return 1 - norm if invert else norm




# loading json
def speechAnalysis(data, y, sr) :
    duration = librosa.get_duration(y=y, sr=sr)

    # extracting text features
    words = data["results"]["channels"][0]["alternatives"][0]["words"]

    #inter word pauses
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i-1]["end"]
        if(gap > 0) :
            pauses.append(gap)

    avg_pause = np.mean(pauses) if pauses else 0
    long_pause_ratio = np.mean([p > 0.5 for p in pauses]) if pauses else 0

    # speech rate
    total_words = len(words)
    speech_rate = total_words / duration if duration > 0 else 0

    # speech rate stability per sentence
    sentences = []
    for p in data["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]:
        for s in p["sentences"]:
            sent_duration = s["end"] - s["start"]
            sent_word_count = len(s["text"].split())
            if sent_duration > 0:
                sentences.append(sent_word_count/sent_duration)

    speech_rate_std = np.std(sentences) if sentences else 0
    speech_rate_stability = 1/ (1+speech_rate_std) # lower std = higher stability

    # Filler density
    fillers = {"um", "uh", "like", "you know"}
    filler_count = sum(1 for w in words if w["word"].lower() in fillers)
    filler_density = filler_count / total_words if total_words > 0 else 0

    # pitch 
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    # Energy (RMS loudness)
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energy_std = np.std(rms) if len(rms) > 0 else 0

    # Voicing ratio (percent of frames above energy threshold)
    voice_frames = np.sum(rms > (0.1 * np.max(rms)))  # arbitrary threshold
    voicing_ratio = voice_frames / len(rms) if len(rms) > 0 else 0

    # Smoothness: inverse of derivative variance
    pitch_derivative = np.diff(pitch_values) if len(pitch_values) > 1 else [0]
    pitch_smoothness = 1 / (1 + np.var(pitch_derivative))


    energy_derivative = np.diff(rms) if len(rms) > 1 else [0]
    energy_smoothness = 1 / (1 + np.var(energy_derivative))


    # Normalizing each feature
    APD_norm = normalize(avg_pause, 0, 1, invert=True)
    LPR_norm = normalize(long_pause_ratio, 0, 1, invert=True)
    SR_norm = normalize(speech_rate, 1, 4)  # 1-4 words/sec reasonable range
    SRS_norm = speech_rate_stability  # already inverted measure
    FD_norm = normalize(filler_density, 0, 0.1, invert=True)

    PV_norm = normalize(pitch_std, 0, 100, invert=True)  # pitch variation
    EV_norm = normalize(energy_std, 0, 0.1, invert=True)  # energy variation
    VR_norm = normalize(voicing_ratio, 0.5, 1)  # lower bound 50%
    PS_norm = pitch_smoothness  # already 0-1
    ES_norm = energy_smoothness  # already 0-1

    # Fluency score formula
    fluency_score = (
        0.1*APD_norm + 0.1*LPR_norm + 0.15*SR_norm + 0.1*SRS_norm +
        0.05*FD_norm + 0.1*PV_norm + 0.1*EV_norm +
        0.1*VR_norm + 0.1*PS_norm + 0.1*ES_norm
    )

    fluencyJson = {
        "Fluency Score (0-1)" : f"{fluency_score:.3f}",
        "Fluency Score (%)" : f"{fluency_score*100:.1f}",
        "WPM" : speech_rate*60,
        "Average Pause Duration": avg_pause,
        "Filler Words" : filler_count,
        "Energy / Loudness" : energy_std,
        "Pitch Variation (Prosody)" : pitch_std,
        
    }

    llmreason = reasoning(average_pause_sec=avg_pause, long_pause_ratio=long_pause_ratio, speech_rate_wps=speech_rate, speech_rate_stability= speech_rate_stability, filler_density=filler_density, pitch_variability=pitch_std, energy_variability=energy_std, voicing_ratio=voicing_ratio, pitch_smoothness=pitch_smoothness, energy_smoothness=energy_smoothness, composite_fluency_score=fluency_score)

    combined_dict = {}
    combined_dict.update(fluencyJson)
    combined_dict.update(llmreason)

    return combined_dict