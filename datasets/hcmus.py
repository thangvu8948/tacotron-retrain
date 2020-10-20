from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import pandas as pd
from util import audio
import xlrd

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):

  # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  
  with open(os.path.join(in_dir, 'metadata.txt'), encoding='utf-8') as f:
    for line in f:
      if line[0] != 'S':
        line = line[0:]
      
      parts = line.strip().split(',')
      wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
      if os.path.isfile(wav_path) == True :
        text = parts[1].replace('\t', ', ')
        futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
        index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    text: The text spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path)

  # Compute the linear-scale spectrogram from the wav:
  spectrogram = audio.spectrogram(wav).astype(np.float32)
  n_frames = spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

  # Write the spectrograms to disk:
  spectrogram_filename = 'hcmus-spec-%05d.npy' % index
  mel_filename = 'hcmus-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (spectrogram_filename, mel_filename, n_frames, text)
