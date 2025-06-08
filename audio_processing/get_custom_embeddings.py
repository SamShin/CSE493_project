# Extracts and saves the VGGish embeddings from our custom test data

import os
from audio_analysis import extract_embeddings
import torch
from torch.nn.utils.rnn import pad_sequence

sound_map = {
    'c': (0, 'baby cry'),
    'l': (1, 'baby laugh'),
    'M': (2, 'music'),
    's': (3, 'singing'),
    'C': (4, 'child speech'),
    'm': (5, 'male speech'),
    'f': (6, 'female speech'),
    'L': (7, 'lullaby')
}

if not all(os.path.exists(f) for f in ['data/custom_test_features.pt', 'data/custom_test_labels.pt']):
    file_names = [f for f in os.listdir('data/test')]
    embeddings = []
    labels = []

    for fn in file_names:
        embeddings.append(extract_embeddings('data/test/' + fn, file=True))
        bin_vector = [0] * 8
        for char, (index, description) in sound_map.items():
            if char in fn:
                bin_vector[index] = 1
                
        labels.append(torch.tensor(bin_vector))

    embeddings = pad_sequence(embeddings, batch_first=True)
    labels = torch.stack(labels)

    torch.save(embeddings.float(), 'data/custom_test_features.pt')
    torch.save(labels.float(), 'data/custom_test_labels.pt')