from scipy.io import wavfile
import numpy as np
import librosa
import librosa.display
from scipy.ndimage import gaussian_filter


def q1(audio_path):
    sample_rate, audio = wavfile.read(audio_path)
    fft_result = np.fft.fft(audio)
    amplitude = np.abs(fft_result)
    fft_result[np.argmax(amplitude)] = 0
    fft_result[len(fft_result) - np.argmax(amplitude)] = 0
    ifft_result = np.fft.ifft(fft_result)
    return ifft_result

def q2(audio_path):
    y, sample_rate = librosa.load(audio_path, sr=4000)
    D = librosa.stft(y, n_fft=2000, hop_length=500)

    # spectrogram
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # make mask
    mask = np.ones(D.shape, dtype=np.float)
    mask[293:312, 11:34] = 0
    sigma = 1.3
    mask_smoothed = gaussian_filter(mask, sigma=sigma)
    D = D * mask_smoothed
    y_filter = librosa.istft(D)
    return y_filter
