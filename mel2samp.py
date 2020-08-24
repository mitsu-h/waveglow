# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
import numpy as np
from glob import glob
from scipy.io.wavfile import read
from torch.utils.data import DataLoader

# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        mel = self.get_mel(audio)
        audio = audio / MAX_WAV_VALUE

        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)

class FineTuneMel2Samp(torch.utils.data.Dataset):
    def __init__(self, training_file_path, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        audio_file_path = os.path.join(training_file_path, '*wav*')
        self.audio_files = glob(audio_file_path)
        self.audio_files.sort()
        mel_file_path = os.path.join(training_file_path, '*mel*')
        self.mel_files = glob(mel_file_path)
        self.mel_files.sort()
        self.segment_length = segment_length
        self.hop_length = hop_length

    def __getitem__(self, index):
        audio_filename = self.audio_files[index]
        audio = np.load(audio_filename)
        audio = torch.from_numpy(audio).float()
        mel_filename = self.mel_files[index]
        mel = np.load(mel_filename)
        mel = torch.from_numpy(mel).float()

        #Take segment
        mel_segment = self.segment_length // self.hop_length
        if mel.size(1) >= mel_segment:
            max_mel_start = mel.size(1) - mel_segment
            mel_start = random.randint(0, max_mel_start)
            mel = mel[:,mel_start:mel_start+mel_segment]
            audio = audio[mel_start*self.hop_length:mel_start*self.hop_length+self.segment_length]
            if audio.size(0) < self.segment_length:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
        else:
            mel = torch.nn.functional.pad(mel, (0, mel_segment - mel.size(1)), 'constant').data
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        return (mel, audio)

    def __len__(self):
        return len(self.audio_files)


# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path")
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory')
    parser.add_argument('-t', '--isFineTuning', type=bool, default=False,
                        help='Do you want to take Fine Tuning?')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    if args.isFineTuning:
        data_config = json.loads(data)["finetune_config"]
        mel2samp = FineTuneMel2Samp(**data_config)

        train_loader = DataLoader(mel2samp, num_workers=1, shuffle=False,
                                  batch_size=1,
                                  pin_memory=False,
                                  drop_last=True)
        for i, batch in enumerate(train_loader):
            mel, audio = batch
            print(mel.size())

    else:
        data_config = json.loads(data)["data_config"]
        mel2samp = Mel2Samp(**data_config)

        filepaths = files_to_list(args.filelist_path)

        # Make directory if it doesn't exist
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
            os.chmod(args.output_dir, 0o775)

        for filepath in filepaths:
            audio, sr = load_wav_to_torch(filepath)
            melspectrogram = mel2samp.get_mel(audio)
            filename = os.path.basename(filepath)
            new_filepath = args.output_dir + '/' + filename + '.pt'
            print(new_filepath)
            torch.save(melspectrogram, new_filepath)
