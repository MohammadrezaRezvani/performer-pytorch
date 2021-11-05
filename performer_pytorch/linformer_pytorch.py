import torch 
from torch import nn


def gen_causal_mask(input_size, dim_k, full_attention=False):
    """
    Generates a causal mask of size (input_size, dim_k) for linformer
    Else, it generates (input_size, input_size) for full attention
    """
    if full_attention:
        return (torch.triu(torch.ones(input_size, input_size))==1).transpose(0,1)
    return (torch.triu(torch.ones(dim_k, input_size))==1).transpose(0,1)

class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """
    def __init__(
        self, 
        dim, 
        dropout, 
        E_proj, 
        F_proj, 
        causal_mask, 
        full_attention=False
    ):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:,:,None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:,:,None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1,2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(K.device)
                K = torch.matmul(K, self.E)
            else:
                K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1,2)
            if self.is_proj_tensor:
                self.F = self.F.to(V.device)
                V = torch.matmul(V, self.F)
            else:
                V = self.F(V)
            V = V.transpose(1,2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

class MHAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """
    def __init__(
        self, 
        dim, 
        nhead,
        dropout, 

        input_size, 
        channels, 
        
        dim_k,                              # necessary fot linformer
        E_proj,                             # necessary fot linformer
        F_proj,                             # necessary fot linformer

        # Don't change default
        w_o_intermediate_dim=None,          # necessary fot linformer
        decoder_mode=False,                 # necessary fot linformer
        method="learnable"                  # necessary fot linformer
    ):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.channels = channels
        self.w_o_intermediate_dim = w_o_intermediate_dim

        self.decoder_mode = decoder_mode
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        # Maybe change causal?
        full_attention = ? # TODO
        self.causal_mask = gen_causal_mask(input_size, dim_k, full_attention) if True else None


        for _ in range(nhead):
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj, self.causal_mask)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias=False))
            self.to_k.append(nn.Linear(channels, dim, bias=False))
            self.to_v.append(nn.Linear(channels, dim, bias=False))
        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim*nhead, channels)
        else:
            self.w_o_1 = nn.Linear(dim*nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.mh_dropout = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape
        assert not (self.decoder_mode and "embeddings" not in kwargs), "Embeddings must be supplied if decoding"
        assert not ("embeddings" in kwargs and (kwargs["embeddings"].shape[0], kwargs["embeddings"].shape[1], kwargs["embeddings"].shape[2]) != (batch_size, input_len, channels)), "Embeddings size must be the same as the input tensor"
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor) if not self.decoder_mode else self.to_k[index](kwargs["embeddings"])
            V = self.to_v[index](tensor) if not self.decoder_mode else self.to_v[index](kwargs["embeddings"])
            
            head_outputs.append(head(Q,K,V,**kwargs))
            
        out = torch.cat(head_outputs, dim=-1)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        out = self.mh_dropout(out)
        return out


##################################
# Linformer
##################################
def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
        return conv
    if method == "no_params":
        mat = torch.zeros((input_size, dim))
        torch.nn.init.normal_(mat, mean=0.0, std=1/dim)
        return mat
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

class linformerAttention(nn.Module):
    def __init__(
        self, 
        dim,
        dropout,
        
        # TODO: Figure out what are these
        input_size,
        
        dim_k = 20,                 # Probably 20? Maybe the dimantion we want K and V be       
        full_attention = False,     # If False it will use linformer implementation
        parameter_sharing = none,   # The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
    ):
        super().__init__()

        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.dim_k = dim_k
        self.full_attention = full_attention

        self.E = get_EF(input_size, dim_k = self.dim_k, method = "learnable", dim = self.dim)
        self.F = get_EF(input_size, dim_k = self.dim_k, method = "learnable", dim = self.dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj

        self.is_proj_tensor = isinstance(self.E, torch.Tensor)

    def forward(self, q, k, v):
        
        k = k.transpose(1,2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(k.device)
                k = torch.matmul(k, self.E)
            else:
                k = self.E(k)
        
        q = torch.matmul(q, K)
        P_bar = q/torch.sqrt(torch.tensor(self.dim).type(q.type())).to(q.device)

        P_bar = P_bar.softmax(dim=-1)
        P_bar = self.dropout(P_bar)
        
        if not self.full_attention:
            v = v.transpose(1,2)
            if self.is_proj_tensor:
                self.F = self.F.to(v.device)
                v = torch.matmul(v, self.F)
            else:
                v = self.F(v)
            v = V.transpose(1,2)
        
        out = torch.matmul(P_bar, V)
        return out