# Extracts and classifies audio segments

import json
from yt_dlp import YoutubeDL
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Retrieve the video links from json file
def getVideoLinks() -> list[str]:
    with open('data/videos.json') as f:
        data = json.load(f)
        return [video['link'] for video in data[:5]]

URLs = getVideoLinks()

ydl_config = {
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }],
    'outtmpl': 'data/dl_videos/%(title)s.%(ext)s',
}

with YoutubeDL(ydl_config) as ydl:
    error_code = ydl.download(URLs)


# Load the audio file (m4a supported via ffmpeg backend)
file_path = 'data/dl_videos/Baby hears music for first time and has beautiful reaction 🥹.m4a'
y, sr = librosa.load(file_path, sr=None)  # sr=None keeps original sample rate
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
log_mel = librosa.power_to_db(mel, ref=np.max)

print(log_mel)
# # Compute spectrogram
# S = librosa.stft(y)
# S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

# # Plot the spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (dB)')
# plt.tight_layout()
# plt.savefig('spectrogram.png')  # Save to file
# plt.show()
