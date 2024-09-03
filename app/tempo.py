import numpy as np
import sounddevice as sd
import librosa
import time
import tkinter as tk

def get_tempo(duration=30, display=True):
    def calculate_tempo(audio, sr):
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo[0]

    # Capturer l'audio pendant la durée spécifiée
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()

    # Calculer le tempo
    tempo = calculate_tempo(recording.flatten(), 44100)

    if display:
        # Créer l'interface graphique pour afficher le tempo
        root = tk.Tk()
        root.title("Tempo Tracker")
        label = tk.Label(root, text=f"Tempo: {tempo:.2f} BPM", font=("Helvetica", 24))
        label.pack(padx=20, pady=20)
        root.mainloop()
    else:
        return tempo

# Exemple d'utilisation
if __name__ == "__main__":
    get_tempo(display=True)