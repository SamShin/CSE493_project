# Portions of this code were sourced or adapted from:
# https://github.com/tensorflow/models/tree/master/research/audioset/vggish
# Copyright 2017 The TensorFlow Authors All Rights Reserved. License: Apache License, Version 2.0 (the "License")

import numpy as np
import torch
import tensorflow.compat.v1 as tf
from vggish_model import mel_features, vggish_input, vggish_params, vggish_postprocess, vggish_slim
import subprocess
import io
from pydub import AudioSegment
from yt_dlp import YoutubeDL
import torchaudio


checkpoint = 'audio_processing/vggish_model/vggish_model.ckpt'
pca_params = 'audio_processing/vggish_model/vggish_pca_params.npz'


# Return the audio wav AudioSegment for a YouTube url
def get_audio_wav(url: str) -> AudioSegment:
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'noplaylist': True,
        'extract_flat': False,
        'forceurl': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        audio_url = info['url']

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', audio_url,
        '-f', 'wav',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',  # downsample to 16 kHz
        '-ac', '1',      # mono
        'pipe:1'         # output to stdout
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wav_bytes, _ = process.communicate()

    return AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")


# Return the post-processed extracted VGGish embeddings for a YouTube url or file path
def extract_embeddings(path: str, file: bool) -> torch.Tensor:
    if file:
        waveform, sr = torchaudio.load(path)
        # Convert to mono if not already
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        
        examples_batch = vggish_input.waveform_to_examples(waveform.numpy()[0], sample_rate=16000)
    else:
        audio_wav = get_audio_wav(path)
        waveform = np.array(audio_wav.get_array_of_samples()).astype(np.float32)
        waveform /= 32768.0
        examples_batch = vggish_input.waveform_to_examples(waveform, sample_rate=16000)

    # Postprocessor to munge model embeddings
    pproc = vggish_postprocess.Postprocessor(pca_params)

    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})
        postprocessed_batch = pproc.postprocess(embedding_batch)

        return torch.from_numpy(postprocessed_batch)