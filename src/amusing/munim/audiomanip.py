from scipy.io.wavfile import read as read_wav
import numpy as np


def open_audio(filename: str) -> np.ndarray:
    sr, audio = read_wav(filename)
    return audio, sr

def fourier(time_domain: np.ndarray) -> np.ndarray:
    return np.fft.fft(time_domain)[:len(time_domain) // 2] / len(time_domain)

def audio2mono_start_end(samples: np.ndarray, start: float = 0.0, end: float = 1.0) -> np.ndarray:
    if len(samples.shape) == 1:
        return samples[round(start * samples.shape[0]):round(end * samples.shape[0]) - 1]
    elif len(samples.shape) == 2:
        return sum(samples[round(start * samples.shape[0]):round(end * samples.shape[0]) - 1, i] for i in range(samples.shape[1])) / samples.shape[1]

def audio2mono_start_length(samples: np.ndarray, start: int, length: int) -> np.ndarray:
    if length >= samples.shape[0]:
        length = samples.shape[0]
    
    if len(samples.shape) == 1:
        return samples[start:start + length]
    elif len(samples.shape) == 2:
        return sum(samples[start:start + length, i] for i in range(samples.shape[1])) / samples.shape[1]
