import torch
import torch.nn as nn
from torch import distributions as D
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import setup_logger

exp_name = "linear_init_contra" 
logger = setup_logger(name="trainer", log_file="/home/anup/AFP3_PB/logs/nf_"+exp_name+".log")


class BimodalGMM(nn.Module):
    # A distribution characterized by Bimodal GMM in each dimension 
    def __init__(self, loc=2.0, std=1.0, num_feats=16, device="cuda"):
        """
        loc: (float), fixed mean of each Gaussian mixture mode. default: +- 2
        std: (float), standard deviation of each Gaussian mixture mode. default: 1
        num_feats: (int), dimension of random variable Z. It is equal to no. of bits.
        """
        super().__init__()
        self.mean = torch.tensor([[-loc, loc]], device=device)
        self.std = torch.tensor([[std, std]], device=device)
        self.wts = torch.tensor([[0.5, 0.5]], device=device)
        self.num_feats= num_feats

    
    def log_prob(self, X):
        """Computes log probability of batch of samples X
        Parameters: 
            X: (float32 tensor), NF output. Shape: B x num_feats (bits)
        Returns:
            log_p: (float32 tensor), log_probability. shape: Bx1
        """
        

        a = -0.5*(torch.pow((X[:,:, None] - self.mean)/self.std, 2))
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        b = torch.log(self.wts) - torch.log(self.std) - torch.log(torch.sqrt(2*torch.tensor(torch.pi)))
        log_p = torch.logsumexp(b+a, dim=-1).sum(-1)
        return log_p
    
    def comp_log_prob(self, X):
        a = -0.5*(torch.pow((X[:,:, None] - self.mean)/self.std, 2))
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        b = - torch.log(self.std) - torch.log(torch.sqrt(2*torch.tensor(torch.pi)))
        return a+b


class Split(nn.Module):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel"):
        """
        The splitting mode can be:

        - channel: Splits first feature dimension, usually channels, into two halfs
        - channel_inv: Same as channel, but with z1 and z2 flipped

        Args:
         mode: splitting mode
        """
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == "channel":
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == "channel_inv":
            z2, z1 = z.chunk(2, dim=1)
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], 1)
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel"):
        super().__init__(mode)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)


class Permute(nn.Module):
    """
    Permutation features along the channel dimension
    """

    def __init__(self, num_channels, mode="shuffle"):
        """Constructor

        Args:
          num_channel: Number of channels
          mode: Mode of permuting features, can be shuffle for random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == "shuffle":
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(self.num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z, context=None):
        if self.mode == "shuffle":
            z = z[:, self.perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det

    def inverse(self, z, context=None):
        if self.mode == "shuffle":
            z = z[:, self.inv_perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : (self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det


class MLP(nn.Module):
    """
    A multilayer perceptron with Leaky ReLU nonlinearities
    """

    def __init__(
        self,
        layers,
        leaky=0.0,
        score_scale=None,
        output_fn=None,
        output_scale=None,
        init_zeros=False,
        dropout=None,
    ):
        """
        layers: list of layer sizes from start to end
        leaky: slope of the leaky part of the ReLU, if 0.0, standard ReLU is used
        score_scale: Factor to apply to the scores, i.e. output before output_fn.
        output_fn: String, function to be applied to the output, either None, "sigmoid", "relu", "tanh", or "clampexp"
        output_scale: Rescale outputs if output_fn is specified, i.e. ```scale * output_fn(out / scale)```
        init_zeros: Flag, if true, weights and biases of last layer are initialized with zeros (helpful for deep models, see [arXiv 1807.03039](https://arxiv.org/abs/1807.03039))
        dropout: Float, if specified, dropout is done before last layer; if None, no dropout is done
        """
        super().__init__()
        net = nn.ModuleList([])
        for k in range(len(layers) - 2):
            net.append(nn.Linear(layers[k], layers[k + 1]))
            net.append(nn.LeakyReLU(leaky))
        if dropout is not None:
            net.append(nn.Dropout(p=dropout))
        net.append(nn.Linear(layers[-2], layers[-1]))
        
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        if output_fn is not None:
            if score_scale is not None:
                net.append(utils.ConstScaleLayer(score_scale))
            if output_fn == "sigmoid":
                net.append(nn.Sigmoid())
            elif output_fn == "relu":
                net.append(nn.ReLU())
            elif output_fn == "tanh":
                net.append(nn.Tanh())
            elif output_fn == "clampexp":
                net.append(utils.ClampExp())
            else:
                NotImplementedError("This output function is not implemented.")
            if output_scale is not None:
                net.append(utils.ConstScaleLayer(output_scale))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z):
        """
        z is a list of z1 and z2; ```z = [z1, z2]```
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1

        Args:
          z
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 + param
            log_det = zero_log_det_like_z(z2)
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 - param
            log_det = zero_log_det_like_z(z2)
        return [z1, z2], log_det


class AffineCouplingBlock(nn.Module):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(self, param_map, scale=True, scale_map="exp", split_mode="channel"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


class NormalizingFlow(nn.Module):

    """
    Normalizing Flow model to approximate target distribution
    """

    def __init__(self, num_layers, nfeats, mlp_units, q0, p=None):
        """Constructor

        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        
        mlp_block = [int(nfeats/2)]
        for units in mlp_units:
            mlp_block.append(units)
        mlp_block.append(nfeats)
        flows = []
        for i in range(num_layers):
            # Neural network with two hidden layers having 64 units each
            # Last layer is initialized by zeros making training more stable
            param_map = MLP(mlp_block, init_zeros=True)
            # Add flow layer
            flows.append(AffineCouplingBlock(param_map))
            # Swap dimensions
            flows.append(Permute(nfeats, mode='swap'))

        self.nfeats = nfeats
        self.flows = nn.ModuleList(flows)
        self.q0 = q0
        self.p = p

    def forward(self, z, return_log_det=False):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian

        Args:
          z: Batch in the latent space

        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d

        if return_log_det:
            return z, log_det
        else:
            return z

    def inverse(self, x, return_log_det=False):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        
        if return_log_det:
            return x, log_det
        else:
            return x

    def forward_kld(self, x):
        """
        Args:
          x: Batch sampled from target distribution

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)

        if -torch.mean(log_q) < 0:
                logger.warning(f"found negative loss. Log q: {log_q}\n############\n")

        return z, -torch.mean(log_q)


