import torch
from torch import nn
import math 

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('bntd,bnm->bmtd',x,A)
        return x.contiguous()
    
class gcn(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear(in_dim,out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, A):
        return self.act(self.dropout(self.mlp(self.nconv(x, A))))

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)  # (1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        b, N, T, d= x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.reshape([b*N, T, d]).contiguous()
        if self.lookup_index is not None:
            x = x + self.pe[:, self.lookup_index, :]  # (batch_size, T, F_in) + (1,T,d_model)
        else:
            x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x.detach()).reshape(b, N, T, -1).contiguous()
    
class STEncoder(nn.Module):
    '''
    input shape: [B,N,T,in_dim]
    output shape: [B,N,T,out_dim]
    '''
    def __init__(self, num_node, in_len, in_dim, out_dim=128):
        super().__init__()
        self.TE = TemporalPositionalEncoding(in_dim, in_len)
        self.mlp = nn.Linear(in_dim, out_dim)
        self.gcn1 = gcn(in_dim, out_dim)
        self.gcn2 = gcn(out_dim, out_dim)
        self.att = nn.MultiheadAttention(out_dim, 1)
        self.norm1 = nn.LayerNorm([num_node, in_len, out_dim])
        self.norm2 = nn.LayerNorm([num_node, in_len, out_dim])
        

    def forward(self, x, A):
        b, n, t = x.shape[0], x.shape[1], x.shape[2]
        x = self.TE(x)
        print(x.shape)
        x1 = self.gcn1(x, A)
        print(x1.shape)
        x1 = x1.reshape(b*n, t, -1).contiguous()
        x2 = self.att(x1, x1, x1)[0]
        x2 = x2.reshape(b, n, t, -1).contiguous()
        x3 = self.norm1(self.mlp(x) + x2)
        x4 = self.gcn2(x3, A)
        x5 = self.norm2(x3 + x4)
        return x5


class NPM(nn.Module):
    def __init__(self, 
                 num_node,
                 in_len,
                 in_dim,
                 out_dim=128, 
                 size=150):
        super().__init__()
        self.encoder = STEncoder(num_node, in_len, in_dim, out_dim)
        self.prompt_bank = nn.Parameter(torch.randn([size, out_dim, out_dim]), requires_grad=True) # B^T B
        self.sqrt = math.sqrt(out_dim)

    def forward(self, X, A):
        Z = self.encoder(X, A) # B, N, T, D
        print(Z.shape, self.prompt_bank.shape)
        P = torch.einsum('bntd,sde->bnte', Z, self.prompt_bank)/self.sqrt
        return torch.cat([X, P], dim=-1), torch.einsum('bntd,bmtd->bnm', P, P)

# if __name__ == '__main__':
#     x = torch.ones(3,2,5,2)
#     a = torch.ones(3, 2, 2)
#     npm = NPM(2, 5, 2, out_dim=2, size=2)
#     p = npm(x, a)
#     print(p)

