import torch 
from torch import nn

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
        input_size,
        
        dim_k = 20,                 # Probably 20? Maybe the dimantion we want K and V be       
        full_attention = False,     # If False it will use linformer implementation
        parameter_sharing = None,   # The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
    ):
        super().__init__()

        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.dim_k = dim_k
        self.full_attention = full_attention
        self.input_size = input_size

        self.print_dim = True
        self.E = get_EF(input_size, dim = self.dim_k, method = "learnable", head_dim = self.dim)
        self.F = get_EF(input_size, dim = self.dim_k, method = "learnable", head_dim = self.dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else self.E

        self.is_proj_tensor = isinstance(self.E, torch.Tensor)

    def forward(self, q, k, v):
        if self.print_dim:
            print("matmul(k, e)")
            print("k:"+str(k.shape))
            print("E:"+str(self.input_size)+", "+str(self.dim_k))
        k = k.transpose(1,2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(k.device)
                k = torch.matmul(k, self.E)
            else:
                k = self.E(k)
        
        if self.print_dim:
            print("matmul(q, k)")
            print("q:"+str(q.shape))
            print("K:"+str(K.shape))
        q = torch.matmul(q, K)
        P_bar = q/torch.sqrt(torch.tensor(self.dim).type(q.type())).to(q.device)

        P_bar = P_bar.softmax(dim=-1)
        P_bar = self.dropout(P_bar)
        
        if not self.full_attention:
            v = v.transpose(1,2)
            if self.is_proj_tensor:
                if self.print_dim:
                    print("matmul(v, F)")
                    print("v:"+str(q.shape))
                    print("F:"+str(self.input_size)+", "+str(self.dim_k))
                    self.print_dim = False
                self.F = self.F.to(v.device)
                v = torch.matmul(v, self.F)
            else:
                v = self.F(v)
            v = V.transpose(1,2)
        
        out = torch.matmul(P_bar, V)
        return out
