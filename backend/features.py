import numpy as np
import librosa

def extract(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, duration = None)  # Conversion of Sound to Numbers (processing).
   
    rms = np.mean(librosa.feature.rms(y=y))         # Soft or Loud Sound.
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]       # Slow or Fast Sound.

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)      # Overall Sound (Timbre).
    mfcc_mean = np.mean(mfcc, axis=1)

    features = np.hstack([rms, tempo, mfcc_mean])       # Combined [rms, temp, mfcc_mean] for easier ML processing.
    return features

'''
 Help from ChatGPT on MFCC, and librosa documentation. If we need realtime audio 
 analysis, we need to use PyAudio, tougher to integrate both the libraries. If 
 we somehow convert audio from microphone to numeric data in PyAudio and then 
 feed it to Librosa for analysis, we can make it happen.

'''