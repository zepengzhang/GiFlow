import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, SimpleConv, ResGatedGraphConv
from .layers import PositionalEmbedding


class SpatialAttention(nn.Module):

    def __init__(self, in_len, hidden_dim, num_layers, dropout):
        super(SpatialAttention, self).__init__()
        
        self.in_lin = nn.Linear(in_len*hidden_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        
        self.gnns = nn.ModuleList([ResGatedGraphConv(hidden_dim, hidden_dim, aggr='mean') for _ in range(num_layers)])
        self.gnn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.q_lin = nn.Linear(hidden_dim, hidden_dim) 
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.out_lin = nn.Linear(hidden_dim, in_len*hidden_dim)
        self.norm_out = nn.LayerNorm(in_len*hidden_dim)
    
    def add_pos_embedding(self, xf):
        B, N, _, _ = xf.shape

        xf = xf.reshape(B, N, -1)
        xf = self.in_lin(xf)
        xf = self.norm_in(xf)
        xf = F.gelu(xf)

        return xf
    
    def compute_A(self, xf):        
        q = self.q_lin(xf)
        k = self.k_lin(xf)
        v = self.v_lin(xf)

        scale = 1. / np.sqrt(q.shape[-1])
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scale * scores

        A = self.dropout(torch.softmax(scores, dim=-1))
        A = A.nan_to_num(0)

        rec = torch.bmm(A, v)
        return rec
    
    def forward(self, xf, edge_index):
        B, N, T, _ = xf.shape

        xf = self.add_pos_embedding(xf=xf)
        rec = self.compute_A(xf=xf)

        rec = self.out_lin(rec)
        rec = self.norm_out(rec)
        rec = rec.reshape(B, N, T, -1)

        return rec


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim, time_dim, dropout=0):
        super(TemporalAttention, self).__init__()

        self.in_lin = nn.Linear(hidden_dim*2, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)

        self.pos_embedding = PositionalEmbedding(d_model=hidden_dim)
        self.time_embedding = nn.Linear(time_dim, hidden_dim)

        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)

        self.out_lin = nn.Linear(hidden_dim, hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)
    
    def add_pos_embedding(self, xf, ex):
        B, N, T, _ = xf.shape

        time_embed = self.time_embedding(ex).unsqueeze(1)

        pos_embed = self.pos_embedding(xf)
        xf = xf + pos_embed

        xf = self.in_lin(torch.cat([
            xf,
            time_embed.repeat(1, N, 1, 1)
        ], dim=-1))
        xf = self.norm_in(xf)
        xf = F.gelu(xf)

        return xf

    def compute_A(self, xf):
        q = self.q_lin(xf)
        k = self.k_lin(xf)
        v = self.v_lin(xf)

        D = q.shape[-1]
        scale = 1. / np.sqrt(D)

        scores = torch.einsum("bnsf, bnft -> bnst", q, k.transpose(2, 3))
        scores = scale * scores

        A = torch.softmax(scores, dim=-1)
        A = A.nan_to_num(0)

        rec = torch.einsum("bnst, bntf -> bnsf", A, v)
        return rec
    
    def forward(self, xf, ex):
        xf = self.add_pos_embedding(xf, ex)
        rec = self.compute_A(xf=xf)
        rec = self.out_lin(rec)
        rec = self.norm_out(rec)
        return rec


class PropagationLayer(nn.Module):
    
    def __init__(self, hidden_dim, in_len, device):
        super(PropagationLayer, self).__init__()

        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.lin_s = nn.Linear(hidden_dim, hidden_dim)
        self.lin_t = nn.Linear(hidden_dim, hidden_dim)

        self.gnn = ResGatedGraphConv(hidden_dim, hidden_dim)

        ones = torch.ones(in_len, in_len).to(device)
        self.t_masking = torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2) \
                       + torch.triu(ones, diagonal=-1) - torch.triu(ones, diagonal=0)
        self.t_masking = self.t_masking / torch.sum(self.t_masking, dim=-1, keepdim=True)

        self.out_lin = nn.Linear(hidden_dim*3, hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

    def t_prop(self, xf):
        xf = self.lin_t(xf)
        t_xf = torch.einsum("st, bntf -> bnsf", self.t_masking, xf)
        return t_xf
    
    def s_prop(self, xf, edge_index):
        xf = self.lin_s(xf)
        xf = xf.transpose(1, 2)
        s_xf = self.gnn(xf, edge_index=edge_index)
        s_xf = s_xf.transpose(1, 2)
        return s_xf
    
    def forward(self, xf, edge_index):
        t_xf = self.t_prop(xf)
        t_xf = self.norm_t(t_xf)

        s_xf = self.s_prop(xf, edge_index=edge_index)
        s_xf = self.norm_s(s_xf)

        xf_lin = self.lin(xf)
        
        fin_xf = self.out_lin(torch.cat([xf_lin, t_xf, s_xf], dim=-1))
        fin_xf = self.norm_out(fin_xf) + xf
        return fin_xf
    
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
    
    def forward(self, x):
        w = x.mean(dim=(1, 2)) 
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)) 
        w = w.view(w.size(0), 1, 1, -1)
        return x * w


class FlowMatching(nn.Module):

    def __init__(self, args):
        super(FlowMatching, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.in_lin = nn.Linear(self.channel, self.hidden_dim)
        self.norm_in = nn.LayerNorm(self.hidden_dim)

        self.in_lin_x0 = nn.Linear(self.channel, self.hidden_dim)
        self.norm_x0 = nn.LayerNorm(self.hidden_dim)

        self.s_attn = SpatialAttention(
            in_len=self.window,
            hidden_dim=self.hidden_dim,
            num_layers=self.spatial_layers,
            dropout=self.dropout)
        
        self.t_attn = TemporalAttention(
            hidden_dim=self.hidden_dim,
            time_dim=self.time_dim,
            dropout=self.dropout)
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.se_block = SEBlock(self.hidden_dim, reduction=16)
        self.stage1_lin = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm_stage1 = nn.LayerNorm(self.hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim*4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.stage2_lin = nn.Linear(self.hidden_dim, self.channel)

        self.stage2 = nn.ModuleList([
            PropagationLayer(hidden_dim=self.hidden_dim, in_len=self.window, device=self.device)
            for _ in range(self.propagation_layers)
        ])

        self.t_s = torch.nn.Parameter(torch.ones(1))
        self.t_t = torch.nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, node_embed, x, x0, ex, edge_index, mask):
        xf_x0 = self.in_lin_x0(x0)
        xf_x0 = self.norm_x0(xf_x0)

        xf = self.in_lin(x)
        xf = self.norm_in(xf)
        xf = F.gelu(xf)
        xf = self.dropout(xf)

        s_rec = self.s_attn(xf=xf, edge_index=edge_index)
        t_rec = self.t_attn(xf=xf, ex=ex[:, :, :4])

        time = ex[:, 0, 4].view(-1, 1, 1, 1).expand(-1, xf.shape[1], xf.shape[2], -1)
        time_emb = self.time_embedding(time)

        fusion = torch.cat([s_rec, t_rec, time_emb, xf_x0], dim=-1)
        fusion = self.fusion_mlp(fusion)
        out = F.gelu(fusion)

        out = self.dropout(out)
        out = self.stage1_lin(out)
        out = self.norm_stage1(out)
        out = F.gelu(out)
        out = self.se_block(out)
        out = self.dropout(out)

        out = torch.where(mask, out, xf)
        for prop in self.stage2:
            out = prop(xf=out, edge_index=edge_index)
            out = F.gelu(out)
            out = torch.where(mask, out, xf)

        out = self.stage2_lin(out)
        return out
