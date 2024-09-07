from calendar import c
from genericpath import isdir
import os
import glob
import pickle
import numpy as np
import torch
import argparse
import yaml
import itertools
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import sys
sys.path.append("../")
from utils.audio import Audio
from utils.features import AudioFeature
from utils import Array
from train import ModelTrainer, ModelPreTrainer
from models import Encoder, NNBlock, NormalizingFlow, BimodalGMM
from tqdm import trange

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


class Indexer():
    def __init__(self,audiopath, ckpt_paths, param_path, fs=16000, audiofeat_params={"n_fft":512, "hop_length":160,"n_mels":64}, device="cpu"):
        self.audiopath = audiopath #(dict), {path1: ext1, path2: ext2,...}. It can store multiple parent <path> and <ext>. <ext> refers to file extension.
        self.ckpt_paths = ckpt_paths #(str), checkpoint path of model weights
        self.audioreader = Audio()
        self.fs = fs #(int, optional) sampling frequency of audio
        self.featextractor = AudioFeature(n_fft=audiofeat_params['n_fft'], hop_length=audiofeat_params['hop_length'], n_mels=audiofeat_params['n_mels'], fs=self.fs)
        self.device=device

        print("Loading files list")
        # read reference filenames to index
        self.files = []
        for k, v in self.audiopath.items():
          self.files.extend(natsorted(glob.glob(os.path.join(k,'**','*.'+v), recursive=True)))

        print("Loading fingerprinter...")
        # load audio fingerprinter
        cfg = pickle.load(open(param_path, "rb"))
        encoder = Encoder(inp_dims=cfg['patch_emb_dim'], patch_size=cfg['patch_size'], nhead=cfg['nhead'], dim_feedforward=cfg['dim_feedforward'], num_layers=cfg['num_layers'],
                        activation=cfg['encoder_activation'], projection_dims=cfg['projection_dims'], concat_position=cfg['concat_position'])
        nnblock = NNBlock(inp_dims=cfg['projection_dims']['out'], bits=cfg['bits'], activation=cfg['activation'], factor=cfg['factor'])
        pretrained_model = ModelPreTrainer.load_from_checkpoint(self.ckpt_paths[0])
        
        if self.device is "cpu":
            base = BimodalGMM(device="cpu")
        else:
            base = BimodalGMM(device="cuda")

        nflows = NormalizingFlow(num_layers=cfg['nf_nblocks'], nfeats=cfg['bits'], mlp_units=cfg['nf_mlp_units'], q0=base)
        self.module = ModelTrainer.load_from_checkpoint(self.ckpt_paths[1], pretrained_model=pretrained_model, nflows=nflows, lambd=cfg['lambda'], temp=cfg['temp'], optimizer=cfg['optimizer'], lr=cfg['lr'], wt_decay=cfg['weight_decay'],)
        self.module.eval()
        if self.device is "cpu":
            self.module.to("cpu")
        else:
            self.module.to("cuda")

        print("initializing index...")
        # placeholder for database and metadata
        self.EMB_DB = Array(10000, 128) ### need to change 128 and 16 depending on the fp dims and bits chosen
        self.PROJ_DB = Array(10000, 16)
        self.Z_DB = Array(10000, 16)
        self.METADATA = Array(10000, 2)
        self.FILES = Array(10, dtype=object)
        self.HASH_TABLE = {}
        self.BAL_HASH_TABLE = {}


    def get_hash_codes(self, batch, threshold):
        """Generate flexible hash codes for batch of z samples depending on the width <threshold>"""
        codes = []
        dis_batch = np.where(batch<0, 0, 1)
        near_zero_indices = np.abs(batch) <= threshold
        num_near_zero_values = near_zero_indices.sum(axis=1)
        for i in range(len(batch)):
            for combination in itertools.product([0, 1], repeat=num_near_zero_values[i]):
                a = np.copy(dis_batch[i])
                a[near_zero_indices[i]] = combination
                codes.append(a)
        return dis_batch, np.array(codes, dtype=int), num_near_zero_values


    def get_fp(self, filepath, hop, threshold):
        """
        Generates cts and discrete fingerprints of an audio track

        Parameters:
            filepath: (str), file path of audio track
            hop: (int(in ms), optional), hop rate to generate fingerprints. default: 100 ms. Note that it should be multiple of 10. 
            threshold: (float, optional), this is the width used to assign both bit 0 and 1 for near-0 z values. 

        Returns:
        """
        # read audio and get spectrograms of overlapping audio chunks of 1s at hop rate of 0.1s
        audio = self.audioreader.read(filepath)
        hop = int(hop*0.1)
        try:
            trimmed_audio = audio[:(audio.shape[0] - (audio.shape[0])%self.fs)]
            spectrum = (self.featextractor.get_log_mel_spectrogram(trimmed_audio)[:, :-1])
            chunks = [spectrum[:,i:i+100] for i in range(0,spectrum.shape[1]-99, hop )]
            chunks = (torch.stack(chunks).unsqueeze(1)).to(self.device)

            # get encoder embeddings(fingeprints) and its corresponding binary encoding
            unnorm_cts_anc_emb, cts_anc_emb, proj_anc_emb, z_anc_emb = self.module.predict_step(chunks, 1) 
            unnorm_cts_anc_emb = unnorm_cts_anc_emb.detach().cpu().numpy()
            cts_anc_emb = cts_anc_emb.detach().cpu().numpy()
            proj_anc_emb = proj_anc_emb.detach().cpu().numpy()
            z_anc_emb = z_anc_emb.detach().cpu().numpy()
            orig_codes, codes, num_near_zero_values = self.get_hash_codes(z_anc_emb, threshold)
            return unnorm_cts_anc_emb, cts_anc_emb, proj_anc_emb, z_anc_emb, orig_codes, codes, num_near_zero_values
        except Exception as e: 
            print(e)
            return -1, -1, -1, -1, -1, -1, -1
    

    def build_index(self, savepath, hop=100, threshold=0.5, N=None):
        """
        Builds reference database and hash table

        Parameters:
            savepath: (str, optional), directory path to store database
            hop: (int(in ms), optional), hop rate to generate fingerprints. default: 100 ms. Note that it should be multiple of 10.
            threshold: (float, optional), this is the width used to assign both bit 0 and 1 for near-0 z values. 
            N: (int, optional), number of files to index. If None, all files will be considered
        """
        if N is not None:
            files = np.random.choice(self.files, N, replace=False)
        else:
            files = self.files

        if os.path.isdir(savepath) is False:
            os.makedirs(savepath)

        file_idx = 0
        for i in trange(len(files)):
            unnorm_cts_anc_emb, cts_anc_emb, proj_anc_emb, z_anc_emb, orig_codes, codes, num_near_zero_values = self.get_fp(files[i], hop, threshold)
            
            try: 
                if isinstance(cts_anc_emb, int) is False:
                    file_idx += 1
                    # build refdbase of fingerprints
                    sz = self.EMB_DB.size
                    self.EMB_DB.add(cts_anc_emb)
                    # self.UNNORM_EMB_DB.add(unnorm_cts_anc_emb)
                    self.PROJ_DB.add(proj_anc_emb)
                    self.Z_DB.add(z_anc_emb)
                    m = np.concatenate([np.ones((len(cts_anc_emb),1))*file_idx, 0.1*(np.arange(0, len(cts_anc_emb)).reshape(-1,1))], axis=1)
                    self.METADATA.add(m)
                    self.FILES.add(files[i])
                
                dum = 0
                for seg_id in range(cts_anc_emb.shape[0]):

                    original_hash_key = int("".join(map(str, list(orig_codes[seg_id]))), 2)
                    if original_hash_key not in self.BAL_HASH_TABLE:
                        self.BAL_HASH_TABLE[original_hash_key] = Array(1000, dtype=np.int32)
                    self.BAL_HASH_TABLE[original_hash_key].add(sz+seg_id)

                    for j in range(2**num_near_zero_values[seg_id]):
                        binary_code = codes[dum+j]
                        hash_key = int("".join(map(str, list(binary_code))), 2)
                        if hash_key not in self.HASH_TABLE:
                            self.HASH_TABLE[hash_key] = Array(1000, dtype=np.int32)
                        self.HASH_TABLE[hash_key].add(sz+seg_id)
                    dum = dum + 2**num_near_zero_values[seg_id]

            except Exception as e:
                print(e)
                continue
        
            if i%1000 == 0:
                print("saving...")
                # pickle.dump(self.UNNORM_EMB_DB, open(os.path.join(savepath, "UNNORM_EMB_DB.pkl"), "wb"), protocol=4)
                pickle.dump(self.EMB_DB, open(os.path.join(savepath, "EMB_DB.pkl"), "wb"), protocol=4)
                pickle.dump(self.PROJ_DB, open(os.path.join(savepath, "PROJ_DB.pkl"), "wb"), protocol=4)
                pickle.dump(self.Z_DB, open(os.path.join(savepath, "Z_DB.pkl"), "wb"), protocol=4)
                pickle.dump(self.FILES, open(os.path.join(savepath, "FILES.pkl"), "wb"), protocol=4)
                pickle.dump(self.METADATA, open(os.path.join(savepath, "METADATA.pkl"), "wb"), protocol=4)
                pickle.dump(self.HASH_TABLE, open(os.path.join(savepath, "HASH_TABLE.pkl"), "wb"), protocol=4)
                pickle.dump(self.BAL_HASH_TABLE, open(os.path.join(savepath, "BAL_HASH_TABLE.pkl"), "wb"), protocol=4)
        return
                        

         
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action="store", required=False, type=str, default="/home/anup/AFP3_PB/config/index.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    indexer = Indexer(cfg['audiopath'], ckpt_paths=cfg['ckpt_paths'], param_path=cfg['param_path'], audiofeat_params=cfg['audiofeat_params'], device="cuda")
    indexer.build_index(savepath=cfg["savepath"], hop=cfg['hop'], threshold=cfg['threshold'])

    
