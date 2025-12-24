import os 
import glob
import torch
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import torch
import torchaudio
from utils.audio import Audio, Augmentations
from utils.features import AudioFeature


files = natsorted(glob.glob("/scratch/sanup/data/FMA/fma_medium/**/*.mp3", recursive=True))
files = np.random.choice(files, 100, replace=False)
noises = natsorted(glob.glob("/scratch/sanup/data/distortions/noise_16k/*.wav"))
rirs = natsorted(glob.glob("/scratch/sanup/data/distortions/rir_16k/*.wav"))

fs = 16000
reader = Audio()
distorter = Augmentations()
feat_extractor = AudioFeature(n_fft=512,hop_length=160, n_mels=64, fs=fs)
basedir = "/scratch/sanup/data/queries"


for file in tqdm(files):
    try:
        audiotrack = reader.read(file)
        offset_with_buffer = np.random.randint(len(audiotrack) - (fs*11)-1)
        noise = reader.read(np.random.choice(noises))
        rir04 = reader.read(rirs[2])
        rir05 = reader.read(rirs[3])

        # GENERATE NOISE AND NOISE REVERB QUERIES
        for snr in range(0,30,5):
            for length in [1,2,3]: #,5,10
                noise_query = distorter.add_noise(audiotrack[offset_with_buffer+fs: offset_with_buffer+fs+(fs*length)], noise, snr)
                # noise_query_spect = feat_extractor.get_log_mel_spectrogram(noise_query)[:,:-1].unsqueeze(0)

                noise_reverb_04_query = distorter.add_noise_reverb(audiotrack[offset_with_buffer:offset_with_buffer+(1+length)*fs], noise, snr, rir04)[fs: (1+length)*fs]
                noise_reverb_05_query = distorter.add_noise_reverb(audiotrack[offset_with_buffer:offset_with_buffer+(1+length)*fs], noise, snr, rir05)[fs: (1+length)*fs]
                # noise_reverb_04_query_spect = feat_extractor.get_log_mel_spectrogram(noise_reverb_04_query)[:,:-1].unsqueeze(0)
                # noise_reverb_05_query_spect = feat_extractor.get_log_mel_spectrogram(noise_reverb_05_query)[:,:-1].unsqueeze(0)
                # print(snr, length, noise_query.shape[0]/fs, noise_query_spect.shape, noise_reverb_04_query.shape[0]/fs, noise_reverb_05_query.shape[0]/fs, noise_reverb_04_query_spect.shape, noise_reverb_05_query_spect.shape)

                query_timeoffset = str((offset_with_buffer + fs)/fs)

                filename = query_timeoffset+"_"+file.split("/")[-2] + "_" +file.split("/")[-1].split('.')[0]
                # print(filename)
                savepath_noise_wav = os.path.join(basedir, "NOISE", str(length), str(snr), filename+".wav")
                # savepath_noise_pt = os.path.join(basedir, "NOISE", str(length), str(snr), filename+".pt")

                savepath_noiserev_04_wav = os.path.join(basedir, "NOISE_REVERB_04", str(length), str(snr), filename+".wav")
                # savepath_noiserev_04_pt = os.path.join(basedir, "NOISE_REVERB_04", str(length), str(snr), filename+".pt")

                savepath_noiserev_05_wav = os.path.join(basedir, "NOISE_REVERB_05", str(length), str(snr), filename+".wav")
                # savepath_noiserev_05_pt = os.path.join(basedir, "NOISE_REVERB_05", str(length), str(snr), filename+".pt")

                if os.path.exists(os.path.dirname(savepath_noise_wav)) is False:
                    os.makedirs(os.path.dirname(savepath_noise_wav))
                
                if os.path.exists(os.path.dirname(savepath_noiserev_04_wav)) is False:
                    os.makedirs(os.path.dirname(savepath_noiserev_04_wav))
                
                if os.path.exists(os.path.dirname(savepath_noiserev_05_wav)) is False:
                    os.makedirs(os.path.dirname(savepath_noiserev_05_wav))

                torchaudio.save(savepath_noise_wav, noise_query.unsqueeze(0), fs)
                torchaudio.save(savepath_noiserev_04_wav, noise_reverb_04_query.unsqueeze(0), fs)
                torchaudio.save(savepath_noiserev_05_wav, noise_reverb_05_query.unsqueeze(0), fs)

                # torch.save(noise_query_spect, savepath_noise_pt)
                # torch.save(noise_reverb_04_query_spect, savepath_noiserev_04_pt)
                # torch.save(noise_reverb_05_query_spect, savepath_noiserev_05_pt)

        # GENERATE REVERB QUERIES
        for rir in rirs:
            t60 = rir.split("/")[-1].split('.wav')[0]
            rirdata = reader.read(rir)
            for length in [1,2,3]: #,5,10
                reverb_query = distorter.add_reverb(audiotrack[offset_with_buffer:offset_with_buffer+ (1+length)*fs], rirdata)[fs: (1+length)*fs]
                # reverb_query_spect = feat_extractor.get_log_mel_spectrogram(reverb_query)[:,:-1].unsqueeze(0)
                query_timeoffset = str((offset_with_buffer + fs)/fs)

                filename = query_timeoffset+"_"+file.split("/")[-2] + "_" +file.split("/")[-1].split('.')[0]
                savepath_reverb_wav = os.path.join(basedir, "REVERB", str(length), t60, filename+".wav")
                # savepath_reverb_pt = os.path.join(basedir, "REVERB", str(length), t60, filename+".pt")
                if os.path.exists(os.path.dirname(savepath_reverb_wav)) is False:
                    os.makedirs(os.path.dirname(savepath_reverb_wav))
                
                torchaudio.save(savepath_reverb_wav, reverb_query.unsqueeze(0), fs)
                # torch.save(reverb_query_spect, savepath_reverb_pt)
    
    except Exception as e:
        print(e) 

