from torch import nn
import torch
from .layers import *
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter
import src.utils as utils
import functools


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class CondEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
        inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        inv = torch.tanh(self.coord_mlp(inv))
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0,
                      reduce='add', dim_size=pos.size(0))
        pos = pos + agg

        return pos


class MultiCondEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim, extra_heads):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        update_heads = 1 + extra_heads
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, update_heads, bias=False)
        )

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb, adj_extra):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        if time_emb is not None:
            shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
            inv = modulate(self.ln(self.input_lin(h_input)), shift, scale)
        else:
            inv = self.ln(self.input_lin(h_input))
        inv = torch.tanh(self.coord_mlp(inv))

        # multi channel adjacency matrix
        adj_dense = torch.ones((adj_extra.size(0), 1), device=adj_extra.device)
        adjs = torch.cat([adj_dense, adj_extra], dim=-1)
        inv = (inv * adjs).mean(-1, keepdim=True)

        # aggregate position
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0,
                      reduce='add', dim_size=pos.size(0))
        pos = pos + agg

        return pos


class EquivariantBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer, without extra heads."""

    def __init__(self, node_dim, edge_dim, time_dim, num_heads, cond_time, dist_gbf, softmax_inf,
                 mlp_ratio=2, act=nn.SiLU(), dropout=0.0, gbf_name='GaussianLayer'):
        super().__init__()


        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.cond_time = cond_time
        self.dist_gbf = dist_gbf
        if dist_gbf:
            dist_dim = edge_dim
        else:
            dist_dim = 1
        self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
        self.node2edge_lin = nn.Linear(node_dim, edge_dim)

        # message passing layer
        self.attn_mpnn = Trans_Layer(node_dim, node_dim // num_heads, num_heads,
                                     edge_dim=edge_dim)

        # Normalization for MPNN
        self.norm1_node = nn.LayerNorm(
            node_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_edge = nn.LayerNorm(
            edge_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        self.norm2_node = nn.LayerNorm(
            node_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> edge.
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)
        self.norm2_edge = nn.LayerNorm(
            edge_dim, elementwise_affine=False, eps=1e-6)

        # equivariant edge update layer
        self.equi_update = CondEquiUpdate(
            node_dim, edge_dim, dist_dim, time_dim)

        self.node_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, node_dim * 6)
        )
        self.edge_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, edge_dim * 6)
        )

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, pos, h, edge_attr, edge_index, node_mask, node_time_emb=None, edge_time_emb=None):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
        """
        h_in_node = h
        h_in_edge = edge_attr
        
        # obtain distance feature
        distance = coord2dist(pos, edge_index)
        if self.dist_gbf:
            distance = self.dist_layer(distance, edge_time_emb)
        edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)

            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            edge_attr = modulate(self.norm1_edge(
                edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr)
        h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + node_gate_msa * \
            h_node if self.cond_time else h_in_node + h_node
        h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
            self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(h_node)) * node_mask

        h_edge = h_in_edge + edge_gate_msa * \
            h_edge if self.cond_time else h_in_edge + h_edge
        h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
            self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(h_edge) if self.cond_time else \
            h_edge + self._ff_block_edge(h_edge)

        # apply equivariant coordinate update
        pos = self.equi_update(h_out, pos, edge_index,
                               h_edge_out, distance, edge_time_emb)
        

        return h_out, h_edge_out, pos


class EquivariantMixBlock(nn.Module):
    """Equivariant block based on graph relational transformer layer."""

    def __init__(self, node_dim, edge_dim, time_dim, num_extra_heads, num_heads, cond_time, dist_gbf, softmax_inf,
                 mlp_ratio=2, act=nn.SiLU(), dropout=0.0, gbf_name='GaussianLayer', trans_name='TransMixLayer'):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.cond_time = cond_time
        self.dist_gbf = dist_gbf
        if dist_gbf:
            dist_dim = edge_dim
        else:
            dist_dim = 1
        self.edge_emb = nn.Linear(edge_dim + dist_dim, edge_dim)
        self.node2edge_lin = nn.Linear(node_dim, edge_dim)

        # message passing layer
        self.attn_mpnn = eval(trans_name)(node_dim, node_dim // num_heads, num_extra_heads, num_heads,
                                          edge_dim=edge_dim, inf=softmax_inf)

        # Normalization for MPNN
        self.norm1_node = nn.LayerNorm(
            node_dim, elementwise_affine=False, eps=1e-6)
        self.norm1_edge = nn.LayerNorm(
            edge_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> node.
        self.ff_linear1 = nn.Linear(node_dim, node_dim * mlp_ratio)
        self.ff_linear2 = nn.Linear(node_dim * mlp_ratio, node_dim)
        self.norm2_node = nn.LayerNorm(
            node_dim, elementwise_affine=False, eps=1e-6)

        # Feed forward block -> edge.
        self.ff_linear3 = nn.Linear(edge_dim, edge_dim * mlp_ratio)
        self.ff_linear4 = nn.Linear(edge_dim * mlp_ratio, edge_dim)
        self.norm2_edge = nn.LayerNorm(
            edge_dim, elementwise_affine=False, eps=1e-6)

        # equivariant edge update layer
        self.equi_update = MultiCondEquiUpdate(
            node_dim, edge_dim, dist_dim, time_dim, num_extra_heads)

        self.node_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, node_dim * 6)
        )
        self.edge_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, edge_dim * 6)
        )

        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)

    def _ff_block_node(self, x):
        x = self.dropout(self.act(self.ff_linear1(x)))
        return self.dropout(self.ff_linear2(x))

    def _ff_block_edge(self, x):
        x = self.dropout(self.act(self.ff_linear3(x)))
        return self.dropout(self.ff_linear4(x))

    def forward(self, pos, h, edge_attr, edge_index, node_mask, extra_heads, node_time_emb=None, edge_time_emb=None):
        """
        Params:
            pos: [B*N, 3]
            h: [B*N, hid_dim]
            edge_attr: [N_edge, edge_hid_dim]
            edge_index: [2, N_edge]
            node_mask: [B*N, 1]
            extra_heads: [N_edge, extra_heads]
        """
        h_in_node = h
        h_in_edge = edge_attr

        # obtain distance feature
        distance = coord2dist(pos, edge_index)
        if self.dist_gbf:
            distance = self.dist_layer(distance, edge_time_emb)
        edge_attr = self.edge_emb(torch.cat([distance, edge_attr], dim=-1))

        # time (noise level) condition
        if self.cond_time:
            node_shift_msa, node_scale_msa, node_gate_msa, node_shift_mlp, node_scale_mlp, node_gate_mlp = \
                self.node_time_mlp(node_time_emb).chunk(6, dim=1)
            edge_shift_msa, edge_scale_msa, edge_gate_msa, edge_shift_mlp, edge_scale_mlp, edge_gate_mlp = \
                self.edge_time_mlp(edge_time_emb).chunk(6, dim=1)

            h = modulate(self.norm1_node(h), node_shift_msa, node_scale_msa)
            edge_attr = modulate(self.norm1_edge(
                edge_attr), edge_shift_msa, edge_scale_msa)
        else:
            h = self.norm1_node(h)
            edge_attr = self.norm1_edge(edge_attr)

        # apply transformer-based message passing, update node features and edge features (FFN + norm)
        h_node = self.attn_mpnn(h, edge_index, edge_attr, extra_heads)
        h_edge = h_node[edge_index[0]] + h_node[edge_index[1]]
        h_edge = self.node2edge_lin(h_edge)

        h_node = h_in_node + node_gate_msa * \
            h_node if self.cond_time else h_in_node + h_node
        h_node = modulate(self.norm2_node(h_node), node_shift_mlp, node_scale_mlp) * node_mask if self.cond_time else \
            self.norm2_node(h_node) * node_mask
        h_out = (h_node + node_gate_mlp * self._ff_block_node(h_node)) * node_mask if self.cond_time else \
                (h_node + self._ff_block_node(h_node)) * node_mask

        h_edge = h_in_edge + edge_gate_msa * \
            h_edge if self.cond_time else h_in_edge + h_edge
        h_edge = modulate(self.norm2_edge(h_edge), edge_shift_mlp, edge_scale_mlp) if self.cond_time else \
            self.norm2_edge(h_edge)
        h_edge_out = h_edge + edge_gate_mlp * self._ff_block_edge(h_edge) if self.cond_time else \
            h_edge + self._ff_block_edge(h_edge)

        # apply equivariant coordinate update
        pos = self.equi_update(h_out, pos, edge_index,
                               h_edge_out, distance, edge_time_emb, extra_heads)
        
        return h_out, h_edge_out, pos


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def coord2diff_adj(x, edge_index, spatial_th=2.):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    with torch.no_grad():
        adj_spatial = radial.clone()
        adj_spatial[adj_spatial <= spatial_th] = 1.
        adj_spatial[adj_spatial > spatial_th] = 0.
    return radial, adj_spatial


def coord2dist(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
    return radial

def from_dense_adj(adj):
    batch_size, max_num_nodes = adj.shape[0], adj.shape[1]
    edge_indices = []
    edge_attrs = []

    for b in range(batch_size):
        edge_index = torch.nonzero(adj[b], as_tuple=False).t()
        edge_attr = adj[b][edge_index[0], edge_index[1]]
        edge_index += b * max_num_nodes  # Adjust for batch offset
        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)
    edge_index = torch.cat(edge_indices, dim=1)
    edge_attr = torch.cat(edge_attrs, dim=0)

    return edge_index, edge_attr
