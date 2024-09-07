import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib import cm


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

    #         # if pl_module.current_epoch > 150:
    #         # nn block embedding(z) similarity matrix
            # figure, ax = plt.subplots(1,1)
            # im = ax.matshow((torch.diag(pl_module.aa.squeeze()) + pl_module.bb).detach().cpu().numpy(), aspect="auto") # L2 dist
            # figure.colorbar(im, shrink=0.8)
            # self.plot_img(figure, "uni_z sim", logger, trainer)

    #         # nn block embedding(z) similarity matrix
    #         figure, ax = plt.subplots(1,1)
    #         im = ax.matshow(pl_module.bb.detach().cpu().numpy(), aspect="auto") # L2 dist
    #         figure.colorbar(im, shrink=0.8)
    #         self.plot_img(figure, "uni_z sim_bb", logger, trainer)

    #         # nn block embedding(z) similarity matrix
    #         figure, ax = plt.subplots(1,1)
    #         im = ax.matshow((torch.diag(pl_module.aa.squeeze())).detach().cpu().numpy(), aspect="auto") # L2 dist
    #         figure.colorbar(im, shrink=0.8)
    #         self.plot_img(figure, "uni_z sim_aa", logger, trainer)

            
            # historgram of distance between positive-anchor and of each anchor/positive from cutoff boundary
            figure, ax = plt.subplots(1,1)
            idx = pl_module.dis_emb_anc != pl_module.dis_emb_pos
            diff_ap = torch.abs(pl_module.z_anc[idx]-pl_module.z_pos[idx]).detach().cpu().numpy() # difference is U-space when bits mismatch, size = (m,)
            # mismatch_per_bit = (dis1!=dis2).sum(0) # no. of mismatches per bit. size: len(bits)=10
            ax.hist(diff_ap, bins=20, edgecolor="black")
            self.plot_img(figure, "z diff histogram", logger, trainer)
            figure, ax = plt.subplots(1,2)
            ax = ax.flatten()
            diff_bd_a = torch.abs(pl_module.z_anc[idx]-pl_module.cutoff).detach().cpu().numpy() #### 0.5 
            diff_bd_p = torch.abs(pl_module.z_pos[idx]-pl_module.cutoff).detach().cpu().numpy() #### 0.5
            ax[0].hist(diff_bd_a, bins=20, edgecolor="black")
            ax[1].hist(diff_bd_p, bins=20, edgecolor="black")
            self.plot_img(figure, "z diff from cutoff boundary histogram", logger, trainer)

            # # buckets found stats
            # figure, ax = plt.subplots(1,1)
            # ax.bar(["ebm", "nbm", "nobm"], 100*pl_module.buckets_found_stats[:3]/np.sum(pl_module.buckets_found_stats[:3]), width=0.4)
            # ax.set_title(str(pl_module.buckets_found_stats[-1]))
            # self.plot_img(figure, "found buckets stats per step", logger, trainer)

            # bits mismatch stats
            figure, ax = plt.subplots(1,1)
            ax.bar(list(pl_module.bits_mismatch_stats.keys()), 100*np.array(list(pl_module.bits_mismatch_stats.values()))/np.sum(list(pl_module.bits_mismatch_stats.values())), width=0.4, edgecolor="k")
            self.plot_img(figure, "bits mismatch stats per step", logger, trainer)


    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        
        # # if pl_module.current_epoch > 70:
        # pl_module.gmm_mu = torch.min(torch.tensor(2.0), pl_module.gmm_mu.clone().detach() + 0.04).to(torch.device("cuda"))
        # pl_module.nflows.basedist.loc = pl_module.gmm_mu
        # print(pl_module.nflows.basedist.loc)
        
        if pl_module.current_epoch % 5 == 0:
            pl_module.IN_DBASE = np.empty((1,pl_module.nflows.q0.num_feats)) ###
            pl_module.DBASE = np.empty((1,pl_module.nflows.q0.num_feats)) ###
            pl_module.DBASE_UNI = np.empty((1,pl_module.nflows.q0.num_feats)) ###
            pl_module.DBASE_DIS = np.empty((1,pl_module.nflows.q0.num_feats))
            pl_module.build_db()
    
            logger = trainer.logger
            axes = np.random.choice(np.arange(0, pl_module.DBASE.shape[1]), (2,), replace=False)
        
            try:
                figure, ax = plt.subplots(1,1)
                im = ax.scatter(pl_module.IN_DBASE[1:, axes[0]], pl_module.IN_DBASE[1:, axes[1]])
                self.plot_img(figure, "input dbase", logger, trainer)

                # nnblock z distribution (standard normal)
                figure, ax = plt.subplots(1,1)
                im = ax.hist2d(pl_module.DBASE[1:, axes[0]], pl_module.DBASE[1:, axes[1]], bins=20, density=False)
                self.plot_img(figure, "dbase (gaussian)", logger, trainer)

                figure, ax = plt.subplots(1,1)
                im = ax.scatter(pl_module.DBASE_UNI[1:, axes[0]], pl_module.DBASE_UNI[1:, axes[1]])
                self.plot_img(figure, "dbase (uniform)", logger, trainer)

                figure, ax = plt.subplots(1,1)
                im = ax.hist2d(pl_module.DBASE_UNI[1:, axes[0]], pl_module.DBASE_UNI[1:, axes[1]], bins=20, density=False)
                self.plot_img(figure, "dbase (uniform) hist2d", logger, trainer)

                # plot correlation matrix
                figure, ax = plt.subplots(1,1)
                im = ax.matshow(np.cov((pl_module.DBASE[1:]).T))
                figure.colorbar(im, shrink=0.8)
                self.plot_img(figure, "correlation matrix", logger, trainer)

                # distribution of z embeddings 
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                B, C = [], []
                for axis in range(pl_module.DBASE_UNI.shape[1]):
                    c, b = np.histogram(pl_module.DBASE_UNI[1:, axis], bins=50, range=[0,1], density=True)
                    C.append(c)
                    B.append((b[:-1]))
                B = np.array(B)
                C = np.array(C)
                x_pos, y_pos = np.meshgrid(B[0], np.arange(pl_module.DBASE_UNI.shape[1]))
                ax.plot_surface(x_pos, y_pos, C, cmap=cm.coolwarm,linewidth=10)
                self.plot_img(fig, "dbase (uniform) hist3d", logger, trainer)

                # distribution of z embeddings 
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                B, C = [], []
                for axis in range(pl_module.DBASE.shape[1]):
                    c, b = np.histogram(pl_module.DBASE[1:, axis], bins=100, range=[-7,7], density=True)
                    C.append(c)
                    B.append((b[:-1]))
                B = np.array(B)
                C = np.array(C)
                x_pos, y_pos = np.meshgrid(B[0], np.arange(pl_module.DBASE.shape[1]))
                ax.plot_surface(x_pos, y_pos, C, cmap=cm.coolwarm,linewidth=10)
                self.plot_img(fig, "dbase (gaussian) hist3d", logger, trainer)
            
            except Exception as e:
                print(e)

            # figure, ax = plt.subplots(1,2)
            # i1_axes = np.argwhere((pl_module.IN_DBASE[:, axes[0]] >= 0) & (pl_module.IN_DBASE[:, axes[1]] >=0))[:,0]
            # i1 = pl_module.IN_DBASE[i1_axes][:, axes]
            # i2_axes = np.argwhere((pl_module.IN_DBASE[:, axes[0]] < 0) & (pl_module.IN_DBASE[:, axes[1]] <0))[:,0]
            # i2 = pl_module.IN_DBASE[i2_axes][:, axes]
            # i3_axes = np.argwhere((pl_module.IN_DBASE[:, axes[0]] >=0) & (pl_module.IN_DBASE[:, axes[1]] <0))[:,0]
            # i3 = pl_module.IN_DBASE[i3_axes][:, axes]
            # i4_axes = np.argwhere((pl_module.IN_DBASE[:, axes[0]] <0) & (pl_module.IN_DBASE[:, axes[1]] >=0))[:,0]
            # i4 = pl_module.IN_DBASE[i4_axes][:, axes]

            # ax[0].scatter(i1[:,0], i1[:, 1])
            # ax[0].scatter(i2[:,0], i2[:, 1])
            # ax[0].scatter(i3[:,0], i3[:, 1])
            # ax[0].scatter(i4[:,0], i4[:, 1])
            # ax[1].scatter(pl_module.DBASE[i1_axes,0], pl_module.DBASE[i1_axes, 1])
            # ax[1].scatter(pl_module.DBASE[i2_axes,0], pl_module.DBASE[i2_axes, 1])
            # ax[1].scatter(pl_module.DBASE[i3_axes,0], pl_module.DBASE[i3_axes, 1])
            # ax[1].scatter(pl_module.DBASE[i4_axes,0], pl_module.DBASE[i4_axes, 1])
            # self.plot_img(figure, "local structure", logger, trainer)









