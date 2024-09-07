import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.distributions import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler
from scipy.stats import skewnorm
from matplotlib import figure
from scipy.stats import norm
from torch import distributions as D
from torch.nn.utils import weight_norm
from pytorch_lightning.callbacks import ModelCheckpoint


n_samples = 10000
numfeats = 2
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# X, y = noisy_moons

X,y = datasets.make_blobs(n_samples=n_samples,  n_features=numfeats, centers=4, cluster_std=1)

# X = skewnorm.rvs(10, size=(n_samples,2))
# y = np.ones(n_samples, dtype=np.int8)

# X = np.random.randn(n_samples,numfeats)
# X = StandardScaler().fit_transform(X)
# y = np.ones((X.shape[0],1))
plt.scatter(X[:,0], X[:, 1])

X = np.concatenate((X,y.reshape(-1,1)), axis=1)

class Data(Dataset):
    def __init__(self, data):
        super().__init__()
        self.X = np.asarray(data, dtype=np.float32) 
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # a = self.X[index]
        # p = a + np.random.randn(a.shape[0])*0.2
        # p = p.astype(a.dtype)
        # # p_corr = np.where(p<1, p, a)
        # return [a,p]
                
        return self.X[index]
    
train_dataset = Data(X[:8000])
valid_dataset = Data(X[8000:])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=True)

class ScaleLayer(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = torch.tensor(scale)
        self.register_buffer("output_scale", self.scale)

    def forward(self, x):
        scaled_x = self.scale * x
        return scaled_x

class Split(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        z = torch.cat([z1, z2], dim=1)
        log_det = 0
        return z, log_det

class Merge(Split):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, z):
        z1 = z[:, :self.dims//2]
        z2 = z[:, self.dims//2:]
        z = torch.cat([z2, z1], dim=1)
        log_det = 0
        return z, log_det

    def inverse(self, z):
        z1 = z[:, :(self.dims+1)//2]
        z2 = z[:, (self.dims+1)//2:]
        z = torch.cat([z2, z1], dim=1)
        log_det = 0
        return z, log_det

class BatchNorm(nn.Module):
    def __init__(self, num_feats, eps=1e-10, momentum=0.9):
        super().__init__()
        self.eps = torch.tensor(eps)
        self.momentum = torch.tensor(momentum)
        self.register_buffer("running_mean", torch.zeros((1, num_feats)))
        self.register_buffer("running_var", torch.ones((1, num_feats)))

    def forward(self, z, mode="training"):
        if self.training:
            mean = self.mean
            var = self.var
        else:
            mean = self.running_mean
            var = self.running_var

        z_ = z * (torch.sqrt(var + self.eps)) + mean
        log_det = 0.5*torch.sum(torch.log(var + self.eps), dim=-1)
        return z_, log_det

    def inverse(self, z, mode="training"):
        if self.training:
            self.mean = torch.mean(z, dim=0, keepdim=True)
            self.var = torch.std(z, dim=0, keepdim=True) ** 2
            self.running_mean = (self.momentum) * \
                self.running_mean + (1-self.momentum)*self.mean
            self.running_var = (self.momentum) * \
                self.running_var + (1-self.momentum)*self.var
        else:
            self.mean = self.running_mean
            self.var = self.running_var

        z_ = (z-self.mean)/(torch.sqrt(self.var + self.eps))
        log_det = -0.5*torch.sum(torch.log(self.var + self.eps), dim=-1)
        return z_, log_det

class MLP(nn.Module):
    def __init__(self, 
                layersize,
                leakyrate = 0.0,
                score_scale=None,
                output_fn=None,
                output_scale=None,
                init_zeros=False,
                dropout=False):

        super().__init__()
        network = nn.ModuleList([])
        for k in range(len(layersize)-2):
            # network.append(weight_norm(nn.Linear(layersize[k], layersize[k+1])))
            network.append(nn.Linear(layersize[k], layersize[k+1]))
            # network.append(nn.BatchNorm1d(num_features=layersize[k+1]))
            network.append(nn.LeakyReLU(leakyrate))
        if dropout:
            network.append(nn.Dropout(p=dropout))
        # network.append(weight_norm(nn.Linear(layersize[-2], layersize[-1])))
        network.append(nn.Linear(layersize[-2], layersize[-1]))
        # network.append(nn.BatchNorm1d(num_features=layersize[-1]))

        if init_zeros:
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)

        if output_fn is not None:
            if score_scale is not None:
                network.append(ScaleLayer(score_scale))
                
            if output_fn == "sigmoid":
                network.append(nn.Sigmoid())
            elif output_fn == "relu":
                network.append(nn.ReLU())
            elif output_fn == "tanh":
                network.append(nn.Tanh())
            else:
                raise NotImplementedError("the specified output function is not implemented")
            
            if output_scale is not None:
                network.append(ScaleLayer(output_scale))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)      

class AffineCoupling(nn.Module):
    def __init__(self, affine_net, scale_map="exp"):
        super().__init__()
        self.affine_net = affine_net
        self.scale_map = scale_map

    def forward(self, z):
        z1, z2 = z 
        params = self.affine_net(z1)
        scale = params[:, 0::2, ...]
        shift = params[:, 1::2, ...]
        if self.scale_map == "exp":
            z2 = z2 * torch.exp(scale) + shift
            log_det = torch.sum(scale, dim=-1) #torch.sum(scale, dim=-1)
        elif self.scale_map == "sigmoid":
            scale_ = torch.sigmoid(scale + 2)
            z2 = z2 * scale_ + shift
            log_det = torch.sum(torch.log(scale_), dim=-1)
        else:
            raise NotImplementedError("The provided scale map is not available")
        return [z1, z2], log_det
    
    def inverse(self, z):
        z1, z2 = z
        params = self.affine_net(z1)
        scale = params[:, 0::2, ...]
        shift = params[:, 1::2, ...]
        if self.scale_map == "exp":
            z2 = (z2 - shift) * torch.exp(-scale)
            log_det = -torch.sum(scale, dim=-1) #-torch.sum(scale, dim=-1)
        elif self.scale_map == "sigmoid":
            scale_ = torch.sigmoid(scale + 2)
            z2 = (z2 - shift) / scale_
            log_det = -torch.sum(torch.log(scale_), dim=-1)
        else:
            raise NotImplementedError("The provided scale map is not available")   
        return [z1,z2], log_det

class AffineCouplingBlock(nn.Module):
    def __init__(self, affine_net, scale_map="exp"):
        super().__init__()
        self.block = nn.ModuleList([])
        self.block.append(Split())
        self.block.append(AffineCoupling(affine_net, scale_map))
        self.block.append(Merge())

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for operation in self.block:
            z, log_det = operation(z)
            log_det_tot += log_det
        return z, log_det_tot
    
    def inverse(self,z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.block)-1, -1, -1):
            z, log_det = self.block[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot

class GaussianMixture(nn.Module):
    def __init__(self, num_feats, modes=2, loc=2.0, std_factor=1.0):
        super().__init__()
        self.mix = D.categorical.Categorical(torch.ones(modes, device="cuda"))
        self.num_feats = num_feats
        self.loc = loc
        self.scale = torch.tensor(std_factor*torch.ones(modes,), dtype=torch.float32, device="cuda")
        self.comp = D.normal.Normal(loc=torch.tensor([-self.loc, self.loc], dtype=torch.float32, device="cuda"), scale=self.scale)
        self.bimodal_gauss = D.mixture_same_family.MixtureSameFamily(self.mix, self.comp)
        self.gmm = self.bimodal_gauss.expand((self.num_feats,))

    def forward(self, num_samples=1):
        z = self.gmm.sample((num_samples,))
        log_p = self.gmm.log_prob(z).sum(-1)
        return z, log_p

    def log_prob(self, z):
        log_p = self.gmm.log_prob(z).sum(-1)
        return log_p

class NFModel(pl.LightningModule):
    def __init__(self, basedist, num_layers, nfeats, mlp_units, lr, init_zeros=False, leakyrate=0.0):
        super().__init__()
        self.basedist = basedist
        self.nfeats = nfeats
        flows = []
        mlp_block = [int(nfeats/2)]
        for units in mlp_units:
            mlp_block.append(units)
        mlp_block.append(nfeats)
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = MLP(mlp_block, init_zeros, leakyrate) #output_fn="sigmoid", 
            # Add flow layer
            flows.append(AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(Permute(nfeats))
        self.flows = nn.ModuleList(flows)
        self.lr = lr #####3
        self.LOSS = []#######
        self.Z = np.empty((1,nfeats))#####
        self.Xhat = np.empty((1,nfeats))####3
        self.X_orig = np.empty((1,nfeats))#####333
        self.X_labels = np.empty((1,))#####

    def forward(self, x, return_z=False):
        return self.forward_KL(x, return_z)
    
    def training_step(self, batch, batch_idx):
        batch, batch_labels = batch[:,:-1], batch[:, -1]
        x_orig, z, loss = self(batch, True)

        if torch.isnan(loss):
            loss = torch.tensor(0., device=loss.device, requires_grad=True)

        xhat = self.reconstruct_samples(z).detach().cpu().numpy() # get xhat samples from z
        self.Z = np.append(self.Z, z.detach().cpu().numpy(), axis=0)
        self.Xhat = np.append(self.Xhat, xhat, axis=0)
        self.X_orig = np.append(self.X_orig, x_orig.detach().cpu().numpy(), axis=0)
        self.X_labels = np.append(self.X_labels, batch_labels.detach().cpu().numpy(), axis=0)
        self.LOSS.append(loss.item())
        if batch_idx == 50:
            self.draw([self.Z[1:], self.Xhat[1:], self.X_orig[1:], self.X_labels[1:], self.LOSS])
            self.Z = np.empty((1,self.nfeats))
            self.Xhat = np.empty((1,self.nfeats)) 
            self.X_orig = np.empty((1,self.nfeats))
            self.X_labels = np.empty((1,))
        self.log("train_loss", loss, on_step=True,on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch, batch_labels = batch[:,:-1], batch[:, -1]
        loss = self(batch)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def forward_KL(self, x, return_z=False):
        z = x
        x_orig = x
        tot_log_det = torch.zeros(x.shape[0], device=x.device)
        for i in range(len(self.flows)-1, -1, -1):
            z, log_det = self.flows[i].inverse(z)  
            tot_log_det += log_det
        
        if torch.sum(~torch.isfinite(z)) > 0:
            log_q = torch.tensor(0)
            # logger.warning(f"found non finite values: {z}")
        else:
            log_q = self.basedist.log_prob(z)

        if return_z:
            return x_orig, z, -torch.mean(log_q + tot_log_det)
        else: 
            return -torch.mean(log_q  + tot_log_det)

    def log_prob(self, x):
        z = x
        tot_log_det = torch.zeros(x.shape[0], device=x.device)
        for i in range(len(self.flows)-1, -1, -1):
            z, log_det = self.flows[i].inverse(z)  
            tot_log_det += log_det
        if torch.sum(~torch.isfinite(z)) > 0:
            log_q = torch.tensor(0)
            # logger.warning(f"found non finite values: {z}")
        else:
            log_q = self.basedist.log_prob(z)
        return log_q

    def sample(self, num_samples=1):
        z, log_q = self.basedist(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            # log_q -= log_det
        return  z#, log_q
    
    def reconstruct_samples(self, z):
        for flow in self.flows:
            z, log_det = flow(z)
        return z
    
    def draw(self, data):
        Z, Xhat, X_orig, X_labels, LOSS = data

        nfeats = Z.shape[-1]
        U = np.zeros_like(Z)
        for feat in range(nfeats):
            U[:,feat] = norm.cdf(Z[:, feat])

        fig = figure.Figure()
        ax = fig.subplots(1,3)
        fig.set_size_inches(12,4)
        ax = ax.flatten()


        self.basedist.comp.log_prob(torch.tensor(Z[...,None]))[:2], 
        best_mode_per_dim = self.basedist.comp.log_prob(torch.tensor(Z[...,None])).max(-1)[1]
        rand_dims = np.random.choice(np.arange(nfeats), (2,), replace=False)
        best_mode_per_dim_sub = best_mode_per_dim[:,rand_dims]
        unique_hash_codes = torch.unique(best_mode_per_dim_sub, dim=0)
        ax[0].scatter(X_orig[:,rand_dims[0]], X_orig[:,rand_dims[1]])
        for i, unique in enumerate(unique_hash_codes):
            idx = torch.where(((unique == best_mode_per_dim_sub).all(-1)) == True)[0]
            ax[1].scatter(Z[idx,rand_dims[0]], Z[idx,rand_dims[1]], label=f"cluster: {i+1}")        
        # ax[1].legend()


        ax[0].set_title("$p_Y$")
        ax[0].set_xlabel("$y_1$")
        ax[0].set_ylabel("$y_2$")

        ax[1].set_title("$p_Z$")
        ax[1].set_xlabel("$z_1$")
        ax[1].set_ylabel("$z_2$")
       
        im = ax[2].hist2d(Z[:,rand_dims[0]], Z[:,rand_dims[1]], bins=20, density=True)
        cbar = fig.colorbar(im[3], ax=ax[2])
        cbar.set_label("density")
        ax[2].set_title("Histogram of z samples")
        ax[2].set_xlabel("$z_1$")
        ax[2].set_ylabel("$z_2$")
        
        fig.tight_layout()
        
        fig.savefig("test.jpg", dpi=400)
        fig.clf()
        plt.close(fig)

        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
    
gmm = GaussianMixture(num_feats=numfeats)
model = NFModel(gmm, num_layers=10, nfeats=numfeats, mlp_units=[64,64,64], lr=1e-4)

callback = ModelCheckpoint(dirpath="./checkpoints/test", filename="{epoch}-{validation_loss:.2f}", save_last=True)
trainer = Trainer(accelerator="gpu",  devices=1, callbacks=callback)
trainer.fit(model, train_loader, valid_loader)