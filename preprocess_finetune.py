"""Prepare training waveform as npy file and dv3 output melspectrogram.

usage: preprocess_finetune.py <in_dir> <out_dir> [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt
import sys
sys.path.insert(0,'deepvoice3')

import numpy as np
import os
from tqdm import tqdm
from os.path import dirname, join, basename, splitext
import librosa

#dv3
import torch
from deepvoice3.deepvoice3_pytorch import frontend
from deepvoice3.hparams import hparams
import deepvoice3.training_module as tm
from deepvoice3.training_module import TextDataSource, MelSpecDataSource
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from deepvoice3.synthesis import _load

_frontend = None
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def save_wav_as_npy(in_dir, out_dir):
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            if len(text) < hparams.min_text:
                continue
            wav = librosa.load(wav_path)[0]
            wav = wav.astype(np.float32)
            wav_filename = 'ljspeech-wav-%05d.npy' % index
            np.save(os.path.join(out_dir, wav_filename), wav, allow_pickle=False)
            index += 1


def save_dv3_mel(data_root, out_dir, checkpoint_path, speaker_id=None):
    #generate dv3 melspec
    # Input dataset definitions
    X = FileSourceDataset(TextDataSource(data_root, speaker_id))
    Mel = FileSourceDataset(MelSpecDataSource(data_root, speaker_id))

    _frontend = getattr(frontend, hparams.frontend)
    import deepvoice3.training_module as tm
    tm._frontend = _frontend

    # Model
    model = tm.build_model()
    checkpoint = _load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])


    for index, (x, mel) in tqdm(enumerate(zip(X, Mel), 1)):
        model.eval()
        model = model.to(device)
        sequence = x
        sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
        text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
        speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)
        r = hparams.outputs_per_step
        max_target_len = len(mel)
        if max_target_len % r != 0:
            max_target_len += r - max_target_len % r
            assert max_target_len % r == 0
        max_target_len += r
        mel = tm._pad_2d(mel, max_target_len, b_pad=r)
        mel = torch.from_numpy(mel).unsqueeze(0).float().to(device)
        frame_positions = torch.arange(1, max_target_len // r + 1).unsqueeze(0).long().to(device)

        # Greedy decoding
        with torch.no_grad():
            mel_outputs, alignments, done = model(
                sequence, mel,  text_positions=text_positions, frame_positions=frame_positions, speaker_ids=speaker_ids)
        mel_spectrogram = mel_outputs[0].cpu().data.numpy()
        mel_filename = 'ljspeech-dv3mel-%05d.npy' % index
        np.save(os.path.join(out_dir, mel_filename), mel_spectrogram[:-r].T, allow_pickle=False)

if __name__ == '__main__':
    args = docopt(__doc__)
    in_dir = args["<in_dir>"]
    out_dir = args["<out_dir>"]
    print("Command line args:\n", args)
    checkpoint_path = args["--checkpoint"]
    speaker_id = args["--speaker-id"]
    speaker_id = int(speaker_id) if speaker_id is not None else None
    preset = args["--preset"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    os.makedirs(out_dir, exist_ok=True)

    #save_wav_as_npy(in_dir, out_dir)
    save_dv3_mel(data_root, out_dir, checkpoint_path)

    print('Finished')