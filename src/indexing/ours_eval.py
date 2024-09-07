import os, sys, pickle, argparse, yaml,torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import sys
sys.path.append("../")
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models import Encoder, NNBlock, NormalizingFlow, BimodalGMM
from train import ModelTrainer, ModelPreTrainer #, Classifier
from utils import MyCallBack, SSLDataset
import numpy as np
import itertools
from tqdm import tqdm, trange
from natsort import natsorted
import glob
from utils.features import AudioFeature
from utils.audio import Audio, Augmentations


# ###################################################################################################
# # SEARCHING AND RECALL EVALUATION
# ###################################################################################################

class Search():
    def __init__(self, ckpt_paths, param_path, dbase, metadata, files, hash_table, fs=16000, featparams={"n_fft":512, "hop_length":160, "n_mels":64}, device="cuda"):
        
        self.ckpt_paths = ckpt_paths
        self.param_path = param_path
        self.dbase = pickle.load(open(dbase, "rb")).getdata()
        self.metadata = pickle.load(open(metadata, "rb")).getdata()
        self.files =  pickle.load(open(files, "rb")).getdata()
        self.hash_table = pickle.load(open(hash_table, "rb"))
        self.fs = fs
        self.device = device
        
        self.featextract = AudioFeature(n_fft=featparams['n_fft'], hop_length=featparams['hop_length'], n_mels=featparams['n_mels'], fs=self.fs) #STFT parameters
        self.extractor = self.featextract.get_log_mel_spectrogram # log Mel feature extractor
        self.audioreader = Audio()

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
        if self.device is "cpu":
            self.module.to("cpu")
        else:
            self.module.to("cuda")
        self.module.eval()
    
    def get_segments(self, audio, hop):  
        if isinstance(audio, str):
            audio = self.audioreader.read(audio)
        hop = int(hop*0.1)
        trimmed_audio = audio[:(audio.shape[0] - (audio.shape[0])%self.fs)]
        spectrum = (self.extractor(trimmed_audio)[:, :-1])
        chunks = [spectrum[:,i:i+100] for i in range(0,spectrum.shape[1]-99, hop )]
        chunks = torch.stack(chunks).unsqueeze(1)
        return  chunks.to(self.device)
    
    def get_hash_codes(self, batch, threshold):
        batch_codes = {}
        dis_batch = torch.where(batch<0, 0, 1).to(torch.int8)
        near_zero_indices = torch.abs(batch) <= threshold
        num_near_zero_values = near_zero_indices.sum(axis=1)
        
        for i in range(len(batch)):
            codes = torch.empty(1,dis_batch.shape[1], dtype=torch.int8, device=batch.device)
            for combination in itertools.product([0, 1], repeat=num_near_zero_values[i]):
                a = dis_batch[i].clone()
                a[near_zero_indices[i]] = torch.tensor(combination, device=a.device, dtype=torch.int8)
                codes = torch.cat((codes, a.unsqueeze(0)), dim=0)   
            batch_codes[i]=codes[1:]
        return batch_codes
    
    def generate_embeddings_codes(self, chunks, threshold):
        _, cts_anc_emb, proj_anc_emb, z_anc_emb = self.module.predict_step(chunks, 1) 
        codes = self.get_hash_codes(z_anc_emb, threshold)
        return cts_anc_emb, proj_anc_emb, z_anc_emb, codes
    
    def codes_to_keys(self, codes):
        batch_probe_keys = {}
        for seg_id in list(codes.keys()):
            probe_keys = []
            for c in codes[seg_id]:
                ht_key = int("".join([str(bit.item()) for bit in c]), 2)
                probe_keys.append(ht_key)
            batch_probe_keys[seg_id] = probe_keys        
        return batch_probe_keys

    def subseq_search(self, retrieved_indices):
        n_segs = len(retrieved_indices)
        retrieved_top1_indices = np.array([retrieved_indices]).T
        a = np.tile(np.arange(n_segs), (n_segs,1)) - np.arange(n_segs).reshape(-1,1)
        Sm = retrieved_top1_indices + a
        Sm, Vm = np.unique(Sm, return_counts=True, axis=0)
        S = Sm[np.argmax(Vm)]
        return S[0], np.max(Vm)
        

    def lookup(self, query, binary_threshold, hop=100, topk=3, tensor_dtype=torch.float32):
        chunks = self.get_segments(query, hop)
        cts_emb, proj_emb, z_emb, codes = self.generate_embeddings_codes(chunks, binary_threshold)
        probe_buckets = self.codes_to_keys(codes)

        eval_cands = 0
        retrieved_top1_indices = []
        for seg_id in range(len(chunks)):
            cands = []
            for bucket in probe_buckets[seg_id]:
                cands.extend(self.hash_table[bucket].getdata().tolist())
            cands, counts = np.unique(cands, return_counts=True)
            cts_emb_cands = torch.from_numpy(self.dbase[cands]).to(self.device).to(tensor_dtype)

            sim_mat = torch.matmul(cts_emb[seg_id].unsqueeze(0).to(tensor_dtype), cts_emb_cands.T)
            sim, idc = torch.topk(sim_mat, topk, dim=-1)
            retrieved_top1_indices.append(cands[idc[0,0]])
            eval_cands+= len(cands)

        idc, evidence = self.subseq_search(retrieved_top1_indices)
        avg_eval_cands = eval_cands/len(chunks)
        avg_eval_cands_perc = 100*avg_eval_cands/self.dbase.shape[0]
        file_match_indicator = int(fname == self.files[self.metadata[idc][0].astype(int) - 1])
        
        if file_match_indicator > 0:
            retrieved_timeoffset = self.metadata[idc][1].item()
        else:
            retrieved_timeoffset = (300)
    
        return file_match_indicator, avg_eval_cands_perc, retrieved_timeoffset, evidence

if __name__ == "__main__":

    FILES = pickle.load(open("/home/anup/AFP3_PB/data/d128/PB_A/BCE_0.3_d128/dbase_files_1.0/FILES.pkl", "rb"))
    print("reading files")
    fs=16000
    reader = Audio()
    distorter = Augmentations()
    featextractor = AudioFeature(n_fft=512,hop_length=160, n_mels=64, fs=fs)
    noises = natsorted(glob.glob("/home/anup/FMA/noise_16k/*.wav"))
    rirs = natsorted(glob.glob("/home/anup/FMA/rir_16k/*.wav"))
    rir04 = reader.read(rirs[2])
    rir05 = reader.read(rirs[3])
    with open("/home/anup/AFP3_PB/config/index.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    searc = Search(ckpt_paths=cfg['ckpt_paths'], param_path=cfg['param_path'], dbase="/home/anup/AFP3_PB/data/d128/PB_A/BCE_0.3_d128/dbase_files_1.0/EMB_DB.pkl", metadata="/home/anup/AFP3_PB/data/d128/PB_A/BCE_0.3_d128/dbase_files_1.0/METADATA.pkl", 
                                                                            files= "/home/anup/AFP3_PB/data/d128/PB_A/BCE_0.3_d128/dbase_files_1.0/FILES.pkl", hash_table="/home/anup/AFP3_PB/data/d128/PB_A/BCE_0.3_d128/dbase_files_1.0/HASH_TABLE.pkl", device="cuda")

    R={}
    length=1
    savepath = "/home/anup/AFP3_PB/data/d128/PB_A/BCE_0.3_d128/dbase_files_1.0/eval/"+str(length)+"sec"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.random.seed(0)
    for snr in tqdm([0, 5, 10, 15, 20]):
    # for rir in rirs:
    #     t60 = rir.split("/")[-1].split('.wav')[0]
    #     rirdata = reader.read(rir)

        FILE_MATCHES = []
        CANDS_PERC = []
        EVIDENCE = []
        T_TRUE = []
        T_RET = []
        fail =0
        for i in trange(1000):
            try:
                fname = FILES.data[np.random.choice(FILES.size),0]
                audiotrack = reader.read(fname)
                offset_with_buffer = np.random.randint(len(audiotrack) - (fs*11)-1)
                noise = reader.read(np.random.choice(noises))
                clean_query = audiotrack[offset_with_buffer+fs: offset_with_buffer+fs+(fs*length)]
                # noise_query = distorter.add_noise(audiotrack[offset_with_buffer+fs: offset_with_buffer+fs+(fs*length)], noise, snr)
                # noise_reverb_05_query = distorter.add_noise_reverb(audiotrack[offset_with_buffer:offset_with_buffer+(1+length)*fs], noise, snr, rir05)[fs: (1+length)*fs]
                # reverb_query = distorter.add_reverb(audiotrack[offset_with_buffer:offset_with_buffer+ (1+length)*fs], rirdata)[fs: (1+length)*fs]
                query_timeoffset = str((offset_with_buffer + fs)/fs)

                # print(fname, query_timeoffset)
                file_match, eval_cands_perc, retrieved_timeoffset, evidence = searc.lookup(noise_reverb_04_query, binary_threshold=1.0, topk=3, tensor_dtype=torch.float32) 
                FILE_MATCHES.append(file_match)
                EVIDENCE.append(evidence)
                CANDS_PERC.append(eval_cands_perc)
                T_TRUE.append(float(query_timeoffset))
                T_RET.append(retrieved_timeoffset)
            except Exception as e:
                print(e)
                fail+=1
                continue
        R[snr] = [FILE_MATCHES, CANDS_PERC, T_RET, T_TRUE, EVIDENCE, fail]
        # R[t60] = [FILE_MATCHES, CANDS_PERC, T_RET, T_TRUE, EVIDENCE, fail]
        pickle.dump(R, open(savepath+"/R_noisereverb_t1.0.pkl", "wb"))


