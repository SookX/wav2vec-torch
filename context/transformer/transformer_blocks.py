import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
        d_k = Q.size(-1)

        transposed_K = K.transpose(-2, -1)
        scores = torch.matmul(Q, transposed_K) / d_k
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, V)

        return output, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads = 4, return_scores = False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.return_scores = return_scores

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.self_att = ScaledDotProductAttention()

    def reshape_head(self, x):
        batch_size, features, _ = x.size()
        x = x.view(batch_size, features, self.n_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    
    def reshape_attention(self, x):
        batch_size, n_heads, features, d_k = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, features, n_heads *d_k)

    def forward(self, x):
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q_h = self.reshape_head(Q)
        K_h = self.reshape_head(K)
        V_h = self.reshape_head(V)

        att, scores = self.self_att(Q_h, K_h, V_h)
        
        att_h = self.reshape_attention(att)
        out = self.o_proj(att_h)

        if self.return_scores:
            return out, scores
        
        return out
    
class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads = 4, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mha = MultiHeadAttention(d_model, n_heads)
    
    def forward(self, x):
        att = self.mha(x)
        att = self.dropout(att)
        return self.layer_norm(x + att)
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        return self.layer_norm(x + ff_out)



if __name__ == "__main__":
    batch_size = 2
    num_heads = 4
    seq_len = 6
    head_dim = 64

    x = torch.randn(batch_size, seq_len, head_dim)

    attention = AttentionBlock(head_dim)
    attention(x)

