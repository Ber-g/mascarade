import numpy as np
import sounddevice as sd
import librosa

def calculate_tempo(audio, sr):
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return tempo[0]

def get_tempo(duration=30):
    # Capturer l'audio pendant la durée spécifiée
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()

    # Calculer le tempo
    tempo = calculate_tempo(recording.flatten(), 44100)

    # Afficher le tempo dans la console
    print(f"Tempo: {tempo:.2f} BPM")

    # Retourner le tempo
    return tempo