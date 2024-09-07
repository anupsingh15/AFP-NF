import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib import cm
import torch.nn as nn


class MyCallBack(pl.Callback):
    def __init__(self):
        super().__init__()

    def plot_img(self, figure,figname,logger,trainer):
        figure.tight_layout()
        figure.canvas.draw()  #dump to memory
        img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,)) #read from memory buffer
        img = img / 255.0 #RGB
        logger.experiment.add_image(figname, img, global_step=trainer.global_step, dataformats='HWC') # add to logger
        plt.close(figure)
        return

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 20 == 0:
            logger = trainer.logger

            # continuous embedding similarity matrix
            figure, ax = plt.subplots(1,1)
            im = ax.matshow(pl_module.temp[0]*np.log(pl_module.sim.detach().cpu().numpy()), aspect="auto") ###cosine sim
            figure.colorbar(im, shrink=0.8)
            self.plot_img(figure, "cts_emb sim", logger, trainer)

            # # continuous embedding similarity matrix
            # figure, ax = plt.subplots(1,1)
            # im = ax.matshow(pl_module.temp[0]*np.log(pl_module.sim1.detach().cpu().numpy()), aspect="auto") ###cosine sim
            # figure.colorbar(im, shrink=0.8)
            # self.plot_img(figure, "proj_emb sim", logger, trainer)

            figure, ax = plt.subplots(1,1)
            im = ax.matshow((torch.diag(pl_module.aa.squeeze()) + pl_module.bb).detach().cpu().numpy(), aspect="auto") # L2 dist
            figure.colorbar(im, shrink=0.8)
            self.plot_img(figure, "proj_emb sim", logger, trainer)

            # figure, ax = plt.subplots(1,1)
            # im = ax.matshow(pl_module.gmm.mean.detach().cpu().numpy(), aspect="auto") ###cosine sim
            # figure.colorbar(im, shrink=0.8)
            # self.plot_img(figure, "gmm_means", logger, trainer)

            # figure, ax = plt.subplots(1,1)
            # im = ax.matshow(pl_module.gmm.std.detach().cpu().numpy(), aspect="auto") ###cosine sim
            # figure.colorbar(im, shrink=0.8)
            # self.plot_img(figure, "gmm_stds", logger, trainer)

            # figure, ax = plt.subplots(1,1)
            # im = ax.matshow(pl_module.gmm.wts.detach().cpu().numpy(), aspect="auto") ###cosine sim
            # figure.colorbar(im, shrink=0.8)
            # self.plot_img(figure, "gmm_wts", logger, trainer)

    # def on_before_zero_grad(self, trainer, pl_module, optimizer):
    #     pl_module.gmm.std.data = pl_module.gmm.std.data.clamp(1, 2)
      

    # @torch.no_grad()
    # def on_train_epoch_end(self, trainer, pl_module):
        
    #     # # # if pl_module.current_epoch > 50:
    #     # pl_module.gmm_mu = torch.min(torch.tensor(2.0), pl_module.gmm_mu.clone().detach() + 0.008).to(torch.device("cuda"))
    #     # pl_module.nflows.basedist.loc = pl_module.gmm_mu
    #     # print(pl_module.nflows.basedist.loc)

        
    #     if pl_module.current_epoch % 5 == 0:
    #         pl_module.IN_DBASE = np.empty((1,16)) ###
    #         pl_module.build_db()
    
    #         logger = trainer.logger
    #         axes = np.random.choice(np.arange(0, pl_module.IN_DBASE.shape[1]), (2,), replace=False)

    #         figure, ax = plt.subplots(1,1)
    #         im = ax.scatter(pl_module.IN_DBASE[1:, axes[0]], pl_module.IN_DBASE[1:, axes[1]])
    #         self.plot_img(figure, "dbase", logger, trainer)

    #         figure, ax = plt.subplots(1,1)
    #         im = ax.hist2d(pl_module.IN_DBASE[1:, axes[0]], pl_module.IN_DBASE[1:, axes[1]], bins=20, density=False)
    #         self.plot_img(figure, "dbase (gaussian)", logger, trainer)


    #         # distribution of z embeddings 
    #         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #         B, C = [], []
    #         for axis in range(pl_module.IN_DBASE.shape[1]):
    #             c, b = np.histogram(pl_module.IN_DBASE[1:, axis], bins=100, range=[-7,7], density=True)
    #             C.append(c)
    #             B.append((b[:-1]))
    #         B = np.array(B)
    #         C = np.array(C)
    #         x_pos, y_pos = np.meshgrid(B[0], np.arange(pl_module.IN_DBASE.shape[1]))
    #         ax.plot_surface(x_pos, y_pos, C, cmap=cm.coolwarm,linewidth=10)
    #         self.plot_img(fig, "dbase (gaussian) hist3d", logger, trainer)


    #         # plot correlation matrix
    #         figure, ax = plt.subplots(1,1)
    #         im = ax.matshow(torch.cov(torch.tensor(pl_module.IN_DBASE[1:]).T))
    #         figure.colorbar(im, shrink=0.8)
    #         self.plot_img(figure, "correlation matrix", logger, trainer)


    #         # distribution of z embeddings 
    #         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #         B, C = [], []
    #         for axis in range(pl_module.DBASE_UNI.shape[1]):
    #             c, b = np.histogram(pl_module.DBASE_UNI[1:, axis], bins=50, range=[0,1], density=True)
    #             C.append(c)
    #             B.append((b[:-1]))
    #         B = np.array(B)
    #         C = np.array(C)
    #         x_pos, y_pos = np.meshgrid(B[0], np.arange(pl_module.DBASE_UNI.shape[1]))
    #         ax.plot_surface(x_pos, y_pos, C, cmap=cm.coolwarm,linewidth=10)
    #         self.plot_img(fig, "dbase (uniform) hist3d", logger, trainer)


    #         # distribution of z embeddings 
    #         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #         B, C = [], []
    #         for axis in range(pl_module.DBASE.shape[1]):
    #             c, b = np.histogram(pl_module.DBASE[1:, axis], bins=100, range=[-7,7], density=True)
    #             C.append(c)
    #             B.append((b[:-1]))
    #         B = np.array(B)
    #         C = np.array(C)
    #         x_pos, y_pos = np.meshgrid(B[0], np.arange(pl_module.DBASE.shape[1]))
    #         ax.plot_surface(x_pos, y_pos, C, cmap=cm.coolwarm,linewidth=10)
    #         self.plot_img(fig, "dbase (gaussian) hist3d", logger, trainer)


            

        


            
        