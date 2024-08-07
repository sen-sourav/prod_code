import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.io import wavfile
import librosa

def save_audio_to_wav(audio_data, filename, sample_rate=16000):
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    if audio_data.ndim == 2:
        audio_data = audio_data.T
    wavfile.write(filename, sample_rate, audio_data)

def int16_to_float32(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

def detect_sound_start(audio, sr, threshold=0.01):
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    energy = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
    start_frame = np.argmax(energy > threshold)
    start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=512)
    return start_time

def my_linear_mixing(audio1, audio2, output_file):
    y1, sr1 = torchaudio.load(audio1)
    y2, sr2 = torchaudio.load(audio2)
    if y2.shape[0] > 1:
        y2 = torch.mean(y2, dim=0).unsqueeze(0)
    if y1.shape[0] > 1:
        y1 = torch.mean(y1, dim=0).unsqueeze(0)
    y2 = torchaudio.functional.resample(y2, sr2, sr1)
    y2 = int16_to_float32(float32_to_int16(y2))
    start_time1 = detect_sound_start(y1, sr1)
    start_time2 = detect_sound_start(y2, sr1)
    time_diff = start_time2 - start_time1
    samples_diff = int(time_diff * sr1)
    if samples_diff > 0:
        y1 = torch.cat([torch.zeros(1, samples_diff), y1], dim=1)
    elif samples_diff < 0:
        y2 = torch.cat([torch.zeros(1, -samples_diff), y2], dim=1)
    max_length = max(y1.shape[1], y2.shape[1])
    if y1.shape[1] < max_length:
        y1 = torch.cat([y1, torch.zeros(1, max_length - y1.shape[1])], dim=1)
    elif y2.shape[1] < max_length:
        y2 = torch.cat([y2, torch.zeros(1, max_length - y2.shape[1])], dim=1)
    y1_norm = (y2.max() / y1.max()) * 0.8
    mixed_audio = (y1 * y1_norm + y2) / y1_norm
    mixed_audio = mixed_audio.squeeze()
    sf.write(output_file, mixed_audio, sr1)
