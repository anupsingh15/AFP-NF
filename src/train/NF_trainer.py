from typing import Any
import torch, os, itertools, glob
import pytorch_lightning as pl
from utils import setup_logger
import numpy as np
from torch.optim import lr_scheduler
from natsort import natsorted
from utils.audio import Audio
from utils.features import AudioFeature
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

exp_name = "NF_proj_cl_bce"
logger = setup_logger(name="trainer", log_file="/home/anup/AFP3_PB/logs/trainer_"+exp_name+".log")


class ModelTrainer(pl.LightningModule):
    def __init__(self,
                pretrained_model,         # (nn.module), Transformer encoder for learning cts audio embeddings
                nflows,
                lambd,
                temp,
                optimizer,       # (str), options: {"adam", "adamw"}
                lr,              # (float), learning rate
                wt_decay=None,   # (float, optional), weight decay parameter when using AdamW. Default: None
                cutoff = 0,
                boundary_thresh = 1
                ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.nflows = nflows
        self.lambd = lambd
        self.temp = temp
        self.optimizer = optimizer
        self.lr = lr
        self.wt_decay = wt_decay
        self.cutoff = cutoff
        self.boundary_thresh = boundary_thresh
        # self.save_hyperparameters()

        self.testfiles = natsorted(glob.glob(os.path.join("/nlsasfs/home/nltm-st/vipular/data/FMA/fma_small/**/*.mp3"), recursive=True))
        self.audioreader = Audio()
        self.featextractor = AudioFeature(n_fft=512, hop_length=160, n_mels=64, fs=16000)
        self.hop = 10

        self.IN_DBASE = np.empty((1,self.nflows.q0.num_feats))
        self.DBASE = np.empty((1,self.nflows.q0.num_feats))
        self.DBASE_UNI = np.empty((1,self.nflows.q0.num_feats))
        self.DBASE_DIS = np.empty((1,self.nflows.q0.num_feats))
        self.bits_mismatch_stats = dict(zip(np.arange(0,self.nflows.q0.num_feats+1), np.zeros(self.nflows.q0.num_feats+1)))
        # self.register_buffer("gmm_mu", torch.tensor(0.0))
    
    def step(self, batch, mode):
        anc, pos = batch

        with torch.no_grad():
            # transformer encodings
            _, cts_anc_emb = self.pretrained_model.encoder(anc) # output: B x D
            _, cts_pos_emb = self.pretrained_model.encoder(pos) 
            # projections
            proj_anc_emb = self.pretrained_model.nnblock(cts_anc_emb) # output: B x d(=bits)
            proj_pos_emb = self.pretrained_model.nnblock(cts_pos_emb)

        # NF embeddings
        self.z_anc, loss_nf = self.nflows.forward_kld(proj_anc_emb) # output: B x d(=bits)
        self.z_pos, _ = self.nflows.forward_kld(proj_pos_emb)
        
        loss = loss_nf 
        
        # mini-batch search                                 
        top1_proj_acc, topk_proj_acc, top1_z_acc, topk_z_acc, dis_match_acc, dis_match_acc_k = self.search(cts_anc_emb, cts_pos_emb, proj_anc_emb, proj_pos_emb)

        self.log(mode+"_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(mode+"_nf_loss", loss_nf, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(mode+"_top1_proj", top1_proj_acc, on_epoch=True, on_step=False, prog_bar=True ,logger=True)
        self.log(mode+"_top3_proj", topk_proj_acc, on_epoch=True, on_step=False, prog_bar=True ,logger=True)
        self.log(mode+"_top1_z", top1_z_acc, on_epoch=True, on_step=False, prog_bar=True ,logger=True)
        self.log(mode+"_top3_z", topk_z_acc, on_epoch=True, on_step=False, prog_bar=True ,logger=True)
        self.log(mode+"_dis_acc", dis_match_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(mode+"_dis_acc_k", dis_match_acc_k, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss 
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")    
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="valid")
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        unnorm_cts_emb, cts_emb,  = self.pretrained_model.encoder(batch)
        proj_emb = self.pretrained_model.nnblock(cts_emb)
        z, _ = self.nflows.forward_kld(proj_emb)
        return unnorm_cts_emb, cts_emb, proj_emb, z
    
    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.nflows.parameters(), lr=self.lr)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-4)
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.nflows.parameters(), lr=self.lr, weight_decay=self.wt_decay)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-4)
        else:
            raise NotImplementedError
        # return {"optimizer": optimizer}
        return {"optimizer": optimizer,  'lr_scheduler':{"scheduler": scheduler, "interval":"epoch"}}
    
    @torch.no_grad()
    def search(self, db_anc, db_pos, proj_anc, proj_pos, mode="train"):
        N = db_anc.shape[0]
        # cutoff = 0
        # boundary_thresh = 1
        topk = 3
        max_bits_mismatch = 3

        # discretize in 2 values: 0/1
        self.dis_emb_anc = torch.where(self.z_anc>self.cutoff,1,0) 
        self.dis_emb_pos = torch.where(self.z_pos>self.cutoff,1,0) 
        
        # # find top-1/k matches on cts embeddings
        # sim_cts = torch.matmul(db_anc, db_pos.t()).to(db_anc.device)
        # _, idc = torch.topk(sim_cts, topk, dim=-1)
        gt_idc = torch.arange(N, device=db_anc.device).view(-1,1)
        # topk_cts_acc = ((gt_idc == idc).count_nonzero())/N
        # top1_cts_acc = ((gt_idc[:,0] == idc[:,0]).count_nonzero())/N
        
        # find top-1/k matches on z embeddings
        sim_z = -torch.linalg.norm(self.z_anc.unsqueeze(1)-self.z_pos, dim=-1).to(db_anc.device)
        _, idc = torch.topk(sim_z, topk, dim=-1)
        topk_z_acc = ((gt_idc == idc).count_nonzero())/N
        top1_z_acc = ((gt_idc[:,0] == idc[:,0]).count_nonzero())/N

        # find top-1/k matches on proj embeddings
        sim_z = -torch.linalg.norm(proj_anc.unsqueeze(1)-proj_pos, dim=-1).to(db_anc.device)
        _, idc = torch.topk(sim_z, topk, dim=-1)
        topk_proj_acc = ((gt_idc == idc).count_nonzero())/N
        top1_proj_acc = ((gt_idc[:,0] == idc[:,0]).count_nonzero())/N

        # find matches for binarized vectors (with no. of bits mismatch)
        dis_match_acc = torch.sum(torch.sum(self.dis_emb_anc == self.dis_emb_pos, dim=-1) >= (self.dis_emb_anc.shape[1]))/N
        dis_match_acc_k = torch.sum(torch.sum(self.dis_emb_anc == self.dis_emb_pos, dim=-1) >= (self.dis_emb_anc.shape[1] - max_bits_mismatch))/N

        
        EXACT_BUCKET_MATCH = 1
        NHBR_BUCKET_MATCH = 1
        NO_BUCKET_MATCH = 1
        BUCKETS_PROBE = []
        found = False
        if mode == "train":
            for i, (dis_ai, dis_pi) in enumerate(zip(self.dis_emb_anc, self.dis_emb_pos)):
                bits_mismatch = self.nflows.q0.num_feats -(dis_ai == dis_pi).sum()
                self.bits_mismatch_stats[bits_mismatch.item()] += 1
        #         bits_flip_cand_index = torch.where((torch.abs(self.z_pos[i] - self.cutoff) < self.boundary_thresh) == True)[0] ############## 0.5, 0.2
        #         BUCKETS_PROBE.append(sum(1 for i in itertools.product([0, 1], repeat=len(bits_flip_cand_index)) if len(i) > 0))
                # if bits_mismatch > 0 and self.current_epoch > 500 and (self.current_epoch%5 == 0) :
                #     # print(f"bits mismatch: {len(bits_flip_cand_index)}\nbits mismatch index: {bits_flip_cand_index}\nanc_dis: {dis_ai}\npos_dis: {dis_pi}")
                #     for bits_comb in itertools.product([0, 1], repeat=len(bits_flip_cand_index)):
                #         dis_q2 = dis_pi.clone()
                #         dis_q2[bits_flip_cand_index] = torch.tensor(bits_comb, device=dis_q2.device)
                #         # print(f"pos_dis: {dis_q2}")
                #         if (self.nflows.q0.num_feats - (dis_q2 == dis_ai).sum()) == 0:
                #             found = True
                #             # print("Yes found")
                #             break
                #     if found:
                #         NHBR_BUCKET_MATCH += 1  
                #     else:
                #         NO_BUCKET_MATCH += 1
                # else:
                #     EXACT_BUCKET_MATCH += 1
                # found=False
        # BUCKETS_PROBE = 100*np.sum(BUCKETS_PROBE)/N
        return top1_proj_acc, topk_proj_acc, top1_z_acc, topk_z_acc, dis_match_acc, dis_match_acc_k#, np.array([EXACT_BUCKET_MATCH, NHBR_BUCKET_MATCH, NO_BUCKET_MATCH, BUCKETS_PROBE])

    @torch.no_grad()
    def build_db(self):
        testfiles = np.random.choice(self.testfiles, 100, replace=False)
        for file in tqdm(testfiles):
            try:
                audio = self.audioreader.read(file)
                trimmed_audio = audio[:(audio.shape[0] - (audio.shape[0])%16000)]
                spectrum = self.featextractor.get_log_mel_spectrogram(trimmed_audio)[:, :-1]
                chunks = [spectrum[:,i:i+100] for i in range(0,spectrum.shape[1]-99, self.hop)]
                if len(chunks) > 1:
                    chunks = torch.stack(chunks).unsqueeze(1)
                    _, _, proj_emb, z = self.predict_step(chunks.to(torch.device("cuda")), 1)
                    self.IN_DBASE = np.concatenate((self.IN_DBASE, proj_emb.detach().cpu().numpy()), axis=0)
                    self.DBASE = np.concatenate((self.DBASE, z.detach().cpu().numpy()), axis=0)
                    uni_z =   0.5 * (1 + torch.erf(z/ torch.sqrt(torch.tensor(2)))) 
                    self.DBASE_UNI = np.concatenate((self.DBASE_UNI, uni_z.detach().cpu().numpy()), axis=0)
                    dis_z = torch.where(uni_z>self.cutoff,1.0,0.0)
                    self.DBASE_DIS = np.concatenate((self.DBASE_DIS, dis_z.detach().cpu().numpy()), axis=0)
            except Exception as e:
                print(e)
        return
    

