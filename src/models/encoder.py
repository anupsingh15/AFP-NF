import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding_Layer(nn.Module):
    """Transforms spectrogram patches into embeddings with their corresponding positions encoded."""
    def __init__(self, patch_size, emb_dims, concat_position=False):
        """
        Parameters:
            patch_size: (tuple, ints), patch size of spectrogram
            emb_dims: (int), linear embedding dims
            concat_position: (boolean, optional): concatenate positional encodings to linear embeddings. If false, adds to linear embeddings
        """
        super().__init__()
        self.patch_size = patch_size
        self.concat_position = concat_position
        self.layer = nn.Conv2d(1, emb_dims, kernel_size=patch_size, stride=patch_size)
    
    def get_positional_encoding(self, n_positions, n_dims, n=10000):
        """
        Parameters:
            n_positions: (int), sequence length
            n_dims: (int), positional encoding dimension
            n: (int), a constant used in generating sinusoidals
        Returns:
            positional_encodings: (float tensor), shape: 1 x <n_positions> x <n_dims> 
        """
        sin_part = np.sin(np.arange(n_positions).reshape(-1,1)/(n**(2*(np.arange(0,n_dims/2).reshape(1,-1))/n_dims)), dtype=np.float32)
        cos_part = np.cos(np.arange(n_positions).reshape(-1,1)/(n**(2*(np.arange(0,np.floor(n_dims/2)).reshape(1,-1))/n_dims)), dtype=np.float32)
        pos_matrix = np.empty((1, n_positions,n_dims), dtype=np.float32)
        pos_matrix[:,:,np.arange(0,n_dims,2)] = sin_part
        pos_matrix[:,:,np.arange(1,n_dims,2)] = cos_part
        return torch.from_numpy(pos_matrix)

    def forward(self, image):
        """
        Parameters:
            image: (float32 tensor), spectrogram. shape: Bx1xFxT
        Returns:
            position encoded spectrogram patch embeddings: B x <n_positions> x <dims>
        """
        if image.dtype != torch.float32:
            raise TypeError(f"Input of type torch.float32 expected. Type {image.dtype} is passed") 
        
        if image.shape[-1] % self.patch_size[1] > 0:
            pad_length = self.patch_size[1]*np.ceil(image.shape[-1]/self.patch_size[1]) - image.shape[-1]
            image = F.pad(image, (0,int(pad_length)))

        x = self.layer(image)
        x = x.view(x.shape[0], -1, x.shape[1])
        pos_enc = self.get_positional_encoding(x.shape[1], x.shape[2])
        pos_enc = pos_enc.to(x.device)

        if self.concat_position:
            out = torch.cat((x, pos_enc.expand(x.shape[0],-1,-1)), dim=-1)
        else:
            out = x + pos_enc
        return out


class Encoder(nn.Module):
    """
    Encoder class. Transforms spectrogram to a sequence of contextual embeddings follwed by their concatenation and projection.
    """
    def __init__(self, inp_dims, patch_size, nhead, dim_feedforward, num_layers, activation, projection_dims, concat_position=False):
        super().__init__()
        """
        Parameters:
            inp_dims: (int), dimension of input embeddings to transformer encoder. Same as dimension of patch embeddings
            patch_size: (tuple, ints), patch size of spectrogram
            nhead: (int), number of attention heads in a self-attention layer
            dim_feedforward: (int), feedforward layer dims. FFN(3-layers): inp_dims -> dim_feedforward -> inp_dims
            num_layers: (int), number of self-attention layers in transformer encoder
            activation: (str), activation used in transformer. {"relu", "gelu"}
            projection_dims: (dict), projection network input and output dims
            concat_position: (boolean, optional): concatenate positional encodings to linear embeddings. If false, adds to linear embeddings
        """
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=inp_dims, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0., activation=activation, layer_norm_eps=1e-05, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.patch_embedding_layer = PatchEmbedding_Layer(patch_size, inp_dims, concat_position)
        self.projection_dims = projection_dims

        self.linear_projector = nn.Sequential()
        self.linear_projector.add_module("l0", nn.Sequential(*[nn.Linear(self.projection_dims['inp'], 2048), nn.BatchNorm1d(2048), nn.ReLU()])) # 2 hidden layers with 2048 neurons
        self.linear_projector.add_module("l1", nn.Sequential(*[nn.Linear(2048, self.projection_dims['out']), nn.BatchNorm1d(self.projection_dims['out'])]))
        
    def forward(self,x,return_ctx=False):
        """
        Parameters:
            x: (float32 tensors), spectrograms, shape: Bx1xFxT
            return_ctx: (bool, optional), return contextual embedding sequence. Default=False
        Returns:
            context_emb (optional): (float32 tensors), contextual continuous embeddings from Transformer encoder. shape: B x T x d, say (Bx10x256) 
            cts_emb: (float32 tensors), projection of concatenated contextual continuous embeddings. shape: B x d', say (Bx128)
        """
        pos_enc = self.patch_embedding_layer(x)
        context_emb = self.encoder(pos_enc)
        cts_emb = self.linear_projector(context_emb.view(context_emb.shape[0], -1))
        if return_ctx:
            return context_emb, cts_emb
        else:
            return cts_emb, F.normalize(cts_emb, dim=-1)


class NNBlock(nn.Module):
    """
    Projects encoder embeddings to smaller dimension (=no. of bits) 
    """
    def __init__(self, inp_dims, bits, activation, factor):
        """
        Parameters:
            inp_dims: (int), input embedding dimensions to feed forward network
            bits: (int), output dimensions
            activation: (str), should be one of {"gelu", "elu", "relu"}
            factor: (int), factor the number of neurons in hidden layers of NN.  
        """
        super().__init__()
        if activation == "gelu":
            activation = nn.GELU()
        elif activation == "elu":
            activation = nn.ELU()
        else:
            activation = nn.ReLU()
        self.hash = nn.Sequential()
        
        # 2 layer FFN as a Quantizers
        self.hash.add_module("l0", nn.Sequential(*[nn.Linear(inp_dims, int(2048/factor[0])), nn.BatchNorm1d(int(2048/factor[0])), activation]))
        self.hash.add_module("l1", nn.Sequential(*[nn.Linear(int(2048/factor[0]), bits)])) # nn.BatchNorm1d(bits)

    def forward(self,x):
        """
        Parameters:
            x: (float32 tensors), contextual continuous embeddings, shape: B x d
        Returns:
            (float32 tensors), output embeddings. shape: B x bits
        """
        # return F.normalize(self.hash(x), dim=-1)
        return self.hash(x)