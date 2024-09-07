from re import I
from torch.utils.data import Dataset
from natsort import natsorted
import glob 
import os
import torch
import numpy as np
import random

from utils.features import AudioFeature
from utils.audio import Audio, Augmentations

class SSLDataset(Dataset):
    """
    A custom dataset class. Return a tuple containing anchor(clean audio), positive(distorted and time shifted audio) sample.
    """
    def __init__(self,
                audiopath,      # (dict): {path1: ext1, path2: ext2,...}. It can store multiple parent paths <path> containing files with <ext> extension.  
                noisepath,      # (list): [path, ext]
                rirpath,        # (list): [path, ext]
                fs,             # (float) sampling rate
                seglen,         # (float) audio segment length in seconds
                power_thresh,   # (float) energy threshold in dB
                audiofeat,      # (str), choices available: {"raw","spectrogam", "logmelspectrogram", "logspectrogram", "melspectrogram"}
                audiofeat_params,# (dict), list of parameters to extract frequency representation
                max_offset,     # (float), max time offset(in seconds) allowed
                snr_range,      # (list), [min_snr(dB), max_snr(dB)]
                distort_probs=None, # (list, optional) a list containing probabilities for each distortion.
                                    # A randomly chosen distortion is added to clean audio. If None, assigns uniform probability over distortions. default=None
                specaug=None):  # (dic), {num_mask: (int), freq_max_width: (float), time_max_width: (float)}. If None, SpecAugment will be not applied.
    
        self.audiopath = audiopath   
        self.noisepath = noisepath 
        self.rirpath = rirpath 
        self.fs = fs 
        self.seglen = seglen 
        self.power_thresh = power_thresh 
        self.audiofeat = audiofeat 
        self.featextract = AudioFeature(n_fft=audiofeat_params['n_fft'], hop_length=audiofeat_params['hop_length'], n_mels=audiofeat_params['n_mels'], fs=self.fs)
        self.featsdic = {"spectrogram": self.featextract.get_spectrogram,
                    "logmelspectrogram": self.featextract.get_log_mel_spectrogram,
                    "logspectrogram": self.featextract.get_log_spectrogram,
                    "melspectrogram": self.featextract.get_mel_spectrogram}
        self.max_offset = max_offset 
        self.snr_range = snr_range 
        self.distort_probs = distort_probs 
        self.specaug = specaug 
        self.audioreader = Audio()
        self.augmenter = Augmentations()

        self.files = []
        for k, v in self.audiopath.items():
          self.files.extend(natsorted(glob.glob(os.path.join(k,'**','*.'+v), recursive=True)))
        self.noises = natsorted(glob.glob(os.path.join(self.noisepath[0], "*."+self.noisepath[1])))
        self.rirs = natsorted(glob.glob(os.path.join(self.rirpath[0], "*."+self.rirpath[1])))

        if len(self.files) == 0:
          raise RuntimeError ("No audio files found. Check path or files extension")
        if len(self.noises) == 0:
            raise RuntimeError ("No noise files found. Check path or files extension")
        if len(self.rirs) == 0:
            raise RuntimeError ("No rirs files found. Check path or files extension")

    def __len__(self):
      return len(self.files)

    def __getitem__(self, idx):
      # clean audio and randomly chosen noise and rir
      audio = self.audioreader.read(self.files[idx])
      if audio is None or torch.sum(torch.isnan(audio)) > 0: 
        # print("audio is none")
        return self.__getitem__(np.random.choice(len(self.files)))
      noise = self.audioreader.read(np.random.choice(self.noises))
      rir = self.audioreader.read(np.random.choice(self.rirs))

      # start indices of anchor and positive samples
      try:
        idx = np.random.choice(len(audio)-  int(self.fs*(1+self.seglen+self.max_offset))) # start index of segment containing buffer
      except:
        # print(len(audio), len(audio)-  int(self.fs*(1+self.seglen+self.max_offset)), "idx problem")
        return self.__getitem__(np.random.choice(len(self.files))) #np.random.choice(len(self.files))

      offset = random.uniform(-self.max_offset, self.max_offset) # time offset in seconds
      start_idx_positive = int(self.fs*(1 + offset)) # start idx of positive sample
      start_idx_anchor = idx + int(self.fs) # start idx of anchor sample

      anchor = audio[start_idx_anchor:start_idx_anchor+int(self.fs*self.seglen)]
      if torch.mean(torch.pow(anchor,2)) <= self.power_thresh:
        # print("power is less", torch.mean(torch.pow(anchor,2)))
        return self.__getitem__(np.random.choice(len(self.files))) #
      anchor_extended = audio[idx:idx + int(self.fs*(1+self.seglen+self.max_offset))]
      audio = None

      # randomly choose type of distortion to add
      choice = np.random.choice([1,2,3], p=self.distort_probs)
      if choice == 1:
        positive = self.augmenter.add_noise(anchor, noise, snr=np.random.choice(np.arange(self.snr_range[0], self.snr_range[1], 5)))
      elif choice == 2:
        positive = self.augmenter.add_reverb(anchor_extended, rir)
        positive = positive[start_idx_positive:start_idx_positive+int(self.fs*self.seglen)]
      elif choice == 3:
        snr=np.random.choice(np.arange(self.snr_range[0], self.snr_range[1], 5))
        positive = self.augmenter.add_noise_reverb(anchor_extended, noise, snr, rir)
        positive = positive[start_idx_positive:start_idx_positive+int(self.fs*self.seglen)]
      
      # return raw waveform or its corresponding frequency domain features
      if self.audiofeat == "raw":
        return anchor, positive
      elif self.audiofeat in self.featsdic.keys():
        anchor_feat = self.featsdic[self.audiofeat](anchor)
        positive_feat = self.featsdic[self.audiofeat](positive)
        if self.specaug is not None:
          postive_feat = self.apply_specAug(positive_feat)     
        return anchor_feat.unsqueeze(0), positive_feat.unsqueeze(0) # spect dim: (1xFxT)
      else:
        raise RuntimeError(f"{self.audiofeat} wrong choice for extracting audio features")
    
    def apply_specAug(self, spect):
        F, T = spect.shape
        # randomly choose masks of random widths
        idxF = np.random.choice(F, self.specaug["num_mask"]).astype(int)
        idxT = np.random.choice(T, self.specaug["num_mask"]).astype(int)
        wF = np.random.uniform(0.05, self.specaug["freq_max_width"], self.specaug["num_mask"])
        wT = np.random.uniform(0.05, self.specaug["time_max_width"], self.specaug["num_mask"])
        wF, wT = np.rint(wF*F).astype(int), np.rint(wT*T).astype(int)
        for i in range(self.specaug["num_mask"]):
            spect[idxF[i]:idxF[i]+wF[i],:] = 0
            spect[:,idxT[i]:idxT[i]+wT[i]] = 0
        return spect


if __name__ == "__main__":
  dataset = SSLDataset(audiopath={"/users/anup/AFP2/data/LibriSpeech/train-clean-100": "flac"},noisepath=["../data/noise_16k/","wav"],  rirpath=["../data/rir_16k/","wav"],
                                fs=16000, seglen=0.95, power_thresh=1e-4, max_offset=0.05, snr_range=[0,30], 
                                audiofeat="logmelspectrogram", audiofeat_params={"n_fft":512, "hop_length":160,"n_mels":64},
                                specaug={"num_mask": 2, "freq_max_width":0.1, "time_max_width":0.1} )
  for i in range(10000):
    print(i)
    anchor, positive = dataset.__getitem__(np.random.randint(len(dataset)))