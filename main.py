import json
import librosa
from topicModelling.topicAnalysis import modelTopic
from SpeechAnalysis.analysis import speechAnalysis

with open('data/saksham.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

topic = "hackathon team formation discussion"
text = data["results"]["channels"][0]["alternatives"][0]["transcript"]

audio_path = "data/audio2.wav"
# y : waveform sr : sample rate
y, sr = librosa.load(audio_path, sr=None) 

topicJson = modelTopic(topic=topic, text=text)
speechJson = speechAnalysis(data=data, y=y, sr=sr)

combined_dict = {}
combined_dict.update(topicJson)
combined_dict.update(speechJson)

print(combined_dict)
print(type(combined_dict))
