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



class ModelPreTrainer(pl.LightningModule):
    def __init__(self,
                encoder,         # (nn.module), Transformer encoder for learning cts audio embeddings
                nnblock,         # projection nn
                temp,            # (list, [temp1, temp2]), Temperature parameters used in contrastive loss. One after encoder level and other after NNblock. 
                optimizer,       # (str), options: {"adam", "adamw"}
                lr,              # (float), learning rate
                wt_decay=None,   # (float, optional), weight decay parameter when using AdamW. Default: None
                ):
        super().__init__()
        self.encoder = encoder
        self.nnblock = nnblock
        self.temp = temp
        self.optimizer = optimizer
        self.lr = lr
        self.wt_decay = wt_decay
        self.save_hyperparameters()
    
    def step(self, batch, mode):
        anc, pos = batch

        # transformer encodings
        _, cts_anc_emb = self.encoder(anc) # output: B x D
        _, cts_pos_emb = self.encoder(pos) 

        proj_anc_emb = self.nnblock(cts_anc_emb)
        proj_pos_emb = self.nnblock(cts_pos_emb)

        # contrast loss on encoder embeddings. cosine similarity used.
        self.sim = torch.exp(torch.matmul(cts_anc_emb, cts_pos_emb.t())/self.temp[0])
        loss_contra = -torch.mean(torch.log(torch.diag(self.sim)/torch.sum(self.sim, dim=-1)))

        # contrast loss on NN block embeddings. L2 norm used.
        self.aa = torch.linalg.norm(proj_anc_emb - proj_pos_emb, dim=-1).unsqueeze(1)
        self.bb = torch.linalg.norm(proj_anc_emb.unsqueeze(1) - proj_anc_emb, dim=-1) + torch.linalg.norm(proj_anc_emb.unsqueeze(1) - proj_pos_emb, dim=-1)
        anc_pos_sim = self.aa
        anc_neg_sim = 0.5*torch.mean(self.bb, dim=-1, keepdim=True)
        loss_contra1 = torch.mean(torch.maximum(torch.tensor(0.0, device="cuda"), anc_pos_sim  + torch.tensor(0.0, device="cuda") - anc_neg_sim)) #6, 4

        top1_cts_acc, topk_cts_acc, top1_proj_acc, topk_proj_acc = self.search(cts_anc_emb, cts_pos_emb, proj_anc_emb, proj_pos_emb)

        # total loss
        loss = loss_contra + loss_contra1 

        self.log(mode+"_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(mode+"_contra_loss", loss_contra, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(mode+"_contra1_loss", loss_contra1, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log(mode+"_top1", top1_cts_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(mode+"_top3", topk_cts_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(mode+"_proj_top1", top1_proj_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        self.log(mode+"_proj_top3", topk_proj_acc, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss 
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")    
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="valid")
    
    def predict_step(self, batch, batch_idx):
        cts_emb = self.encoder(batch)
        proj_emb = self.nnblock(cts_emb)
        return cts_emb, proj_emb
    
    def configure_optimizers(self):
        if self.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wt_decay)
        else:
            raise NotImplementedError
        return {"optimizer": optimizer}  

    @torch.no_grad()
    def search(self, db_anc, db_pos, proj_anc_emb, proj_pos_emb, mode="train"):
        N = db_anc.shape[0]
        topk = 3

        # find top-1/k matches on cts embeddings
        sim_cts = torch.matmul(db_anc, db_pos.t()).to(db_anc.device)
        _, idc = torch.topk(sim_cts,topk, dim=-1)
        gt_idc = torch.arange(N, device=db_anc.device).view(-1,1)
        topk_cts_acc = ((gt_idc == idc).count_nonzero())/N
        top1_cts_acc = ((gt_idc[:,0] == idc[:,0]).count_nonzero())/N


        # find top-1/k matches on z embeddings
        sim_z = -torch.linalg.norm(proj_anc_emb.unsqueeze(1)-proj_pos_emb, dim=-1).to(db_anc.device)
        _, idc = torch.topk(sim_z, topk, dim=-1)
        topk_proj_acc = ((gt_idc == idc).count_nonzero())/N
        top1_proj_acc = ((gt_idc[:,0] == idc[:,0]).count_nonzero())/N

        return top1_cts_acc, topk_cts_acc, top1_proj_acc, topk_proj_acc


