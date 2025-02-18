from typing import Optional,Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_adj

import src.utils as utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy, SE3Norm, PositionsMLP, masked_softmax, EtoX, SetNorm, GraphNorm
from .TGT import *


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, last_layer=False) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, last_layer=last_layer)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        # self.normX1 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        # self.normX2 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps, **kw)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        # self.normE1 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        # self.normE2 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, features: utils.PlaceHolder):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        X = features.X
        E = features.E
        y = features.y
        pos = features.pos
        node_mask = features.node_mask
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        newX, newE, new_y, vel = self.self_attn(X, E, y, pos, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        # X = self.normX1(X + newX_d, x_mask)
        X = self.normX1(X + newX_d)
        # new_pos = pos + vel
        new_pos = self.norm_pos1(vel, x_mask) + pos
        if torch.isnan(new_pos).any():
            raise ValueError("NaN in new_pos")

        newE_d = self.dropoutE1(newE)
        # E = self.normE1(E + newE_d, e_mask1, e_mask2)
        E = self.normE1(E + newE_d)

        if not self.last_layer:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        # X = self.normX2(X + ff_outputX, x_mask)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        # E = self.normE2(E + ff_outputE, e_mask1, e_mask2)
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)

        out = utils.PlaceHolder(X=X, E=E, y=y, pos=new_pos, charges=None, node_mask=node_mask).mask()

        return out


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, last_layer=False):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        # self.x_e_add = Linear(dx, de)
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_norm_pos1 = Linear(1, de)
        self.lin_norm_pos2 = Linear(1, de)

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)
        # self.lin_dist2 = Linear(dx, dx)

        # Attention
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)

        # Incorporate e to x
        # self.e_att_add = Linear(de, n_head)
        self.e_att_mul = Linear(de, n_head)

        self.pos_att_mul = Linear(de, n_head)

        self.e_x_mul = EtoX(de, dx)

        self.pos_x_mul = EtoX(de, dx)


        # FiLM y to E
        self.y_e_mul = Linear(dy, de)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)       # Unused, but needed to load old checkpoints

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)

        # Process_pos
        self.e_pos1 = Linear(de, de, bias=False)
        self.e_pos2 = Linear(de, 1, bias=False)          # For EGNN v3: map to pi, pj

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if not last_layer:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, pos, node_mask):
        """ :param X: bs, n, d        node features
            :param E: bs, n, n, d     edge features
            :param y: bs, dz           global features
            :param pos: bs, n, 3
            :param node_mask: bs, n
            :return: newX, newE, new_y with the same shape. """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 0. Create a distance matrix that can be used later
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)         # bs, n, 1
        normalized_pos = pos / (norm_pos + 1e-7)                 # bs, n, 3

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1).float()
        cosines = torch.sum(normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2), dim=-1, keepdim=True)
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)

        norm1 = self.lin_norm_pos1(norm_pos)             # bs, n, de
        norm2 = self.lin_norm_pos2(norm_pos)             # bs, n, de
        dist1 = F.relu(self.lin_dist1(pos_info) + norm1.unsqueeze(2) + norm2.unsqueeze(1)) * e_mask1 * e_mask2

        # 1. Process E
        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2

        # 1.2. Incorporate distances
        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2   # bs, n, n, dx

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(E) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(Eout, e_mask1 * e_mask2)

        # 2. Process the node features
        Q = (self.q(X) * x_mask).unsqueeze(2)          # bs, 1, n, dx
        K = (self.k(X) * x_mask).unsqueeze(1)          # bs, n, 1, dx
        prod = Q * K / math.sqrt(Y.size(-1))   # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2   # bs, n, n, n_head

        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a

        # 2.2 Incorporate position features
        pos_x_mul = self.pos_att_mul(dist1)
        a = a + pos_x_mul * a
        a = a * e_mask1 * e_mask2

        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        weighted_V = alpha * V                                  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)                      # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, n_head x dx
        weighted_V = self.out(weighted_V) * x_mask              # bs, n, dx

        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V

        pos_x_mul = self.pos_x_mul(dist1, e_mask2)
        weighted_V = weighted_V + pos_x_mul * weighted_V

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)                     # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(Xout, x_mask)

        # Process y based on X and E
        if self.last_layer:
            y_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            new_y = y + x_y + e_y + dist_y
            y_out = self.y_out(new_y)               # bs, dy

        # Update the positions
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)              # bs, 1, n, 3
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)              # bs, n, 1, 3
        delta_pos = pos2 - pos1                                    # bs, n, n, 3

        messages = self.e_pos2(F.relu(self.e_pos1(Y)))       # bs, n, n, 1, 2
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        vel = utils.remove_mean_with_mask(vel, node_mask)
        return Xout, Eout, y_out, vel


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, input_dims: utils.PlaceHolder, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: utils.PlaceHolder,DGT_configs: dict):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges
        n_heads=hidden_dims['n_head']
        self.DGT_n_layer = DGT_n_layers =DGT_configs['n_layer']

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()
        


        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges, hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims.E, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims.y, hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims['pos'])

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=n_heads,
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            last_layer=False)     # needed to load old checkpoints
                                                            # last_layer=(i == n_layers - 1))
                                        for i in range(n_layers)])
        
        cat_node_dim = (hidden_dims['dx'] * 2) // DGT_n_layers
        cat_edge_dim = (hidden_dims['de'] * 2) // DGT_n_layers
        n_extra_heads = DGT_configs['n_extra_heads']
        self.cond_time = cond_time = DGT_configs['cond_time']
        self.dist_gbf = dist_gbf = DGT_configs['dist_gbf']
        softmax_inf = DGT_configs['softmax_inf']
        mlp_ratio = DGT_configs['mlp_ratio']
        self.dropout = dropout =DGT_configs['dropout']
        gbf_name = DGT_configs['gbf_name']
        time_dim = hidden_dims['dx'] * 4
        self.spatial_cut_off = DGT_configs['spatial_cut_off']
        self.CoM = DGT_configs['CoM']
        
        if dist_gbf:
            self.dist_dim = dist_dim= hidden_dims['de']
        else:
            self.dist_dim = dist_dim= 1
        
        in_edge_dim = DGT_configs['edge_ch'] * 2 + dist_dim
        
        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)
        
        hidden_dim = hidden_dims['dx'] 
        edge_hidden_dim = hidden_dims['de']

        in_node_dim = DGT_configs['atom_types'] + \
            int(DGT_configs['include_fc_charge'])
        self.node_emb = nn.Linear(in_node_dim + 2, hidden_dim)
        self.edge_emb = nn.Linear(in_edge_dim, edge_hidden_dim)

        self.node_pred_mlp = nn.Sequential(
            nn.Linear(cat_node_dim * DGT_n_layers + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, in_node_dim + 2)
        )
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * DGT_n_layers +
                      edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, DGT_configs['edge_ch'] - 1)
        )
        self.edge_exist_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * DGT_n_layers +
                      edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, 1)
        )
        
        for i in range(DGT_n_layers):
            self.add_module("e_block_%d" % i, EquivariantMixBlock(node_dim=hidden_dims['dx'], edge_dim=hidden_dims['de'], time_dim=time_dim, num_extra_heads=n_extra_heads,
                            num_heads=n_heads, cond_time=cond_time, dist_gbf=dist_gbf, softmax_inf=softmax_inf, mlp_ratio=mlp_ratio, dropout=dropout,
                            gbf_name=gbf_name, trans_name=DGT_configs['trans_name']))
            self.add_module("node_%d" % i, nn.Linear(hidden_dims['dx'], cat_node_dim))
            self.add_module("edge_%d" % i, nn.Linear(hidden_dims['de'], cat_edge_dim))

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims.X + output_dims.charges))
        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims.E))
        # self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
        #                                nn.Linear(hidden_mlp_dims['y'], output_dims.y))
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims['pos'])
        
    def forward(self, data: utils.PlaceHolder):
        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask

        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        # edge_mask = diag_mask & node_mask.reshape(bs,n).unsqueeze(-1) & node_mask.reshape(bs,n).unsqueeze(-2)
        X = torch.cat((data.X, data.charges), dim=-1)

        # print('X.shape',X.shape)
        # print('E.shape',data.E.shape)
        # print('pos.shape',data.pos.shape)
        # print('node_mask.shape',node_mask.shape)

        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]
        E_to_out = data.E[..., :self.out_dim_E]
        y_to_out = data.y[..., :self.out_dim_y]

        # start DGT
        # init input
        input_x = X.reshape(bs*n, -1).float()
        input_pos = pos = data.pos.clone().reshape(bs * n, -1)

        E_original = torch.argmax(data.E, dim=-1)
        edge_index, input_e = dense_to_sparse(E_original)
        input_e = F.one_hot(input_e, num_classes=5).float()
        edge_x = input_e.float()
        cond_adj_2d = torch.ones(
            (edge_index.size(1), 1), device=edge_x.device)
        distances, cond_adj_spatial = coord2diff_adj(
            data.pos.reshape(bs * n, -1), edge_index, self.spatial_cut_off)
        if distances.sum() == 0:
            distances = distances.repeat(1, self.dist_dim)
        else:

            if self.dist_gbf:
                distances = self.dist_layer(distances, None)
        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        input_e = self.mlp_in_E(input_e)
        input_x = self.mlp_in_X(input_x)

        h = input_x.float()
        edge_attr = input_e.float()

        # run DGT Blocks
        atom_hids = [h]
        edge_hids = [edge_attr]

        # DGT_input = utils.PlaceHolder(pos=input_pos, X=atom_hids, E=edge_hids, charges=None, y=y, node_mask=input_node_mask).mask()

        for i in range(0, self.DGT_n_layer):
           h, edge_attr, pos = self._modules['e_block_%d' % i](
               input_pos.reshape(-1, 3), h, edge_attr, edge_index, node_mask.reshape(-1, 1), extra_adj, node_time_emb=None, edge_time_emb=None)
           if self.CoM:
               pos = my_remove_mean_with_mask(pos.reshape(
                   bs, n, -1), node_mask.reshape(bs, n, 1)).reshape(bs * n, -1)
           # atom_hids.append(self._modules['node_%d' % i](h))
           # edge_hids.append(self._modules['edge_%d' % i](edge_attr))

        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)

        edge_pred = self.mlp_out_E(edge_hids)
        # atom_pred = self.mlp_out_X(atom_hids).reshape(
        #     bs, n, -1) * node_mask.reshape(bs, n, 1)

        # atom_pred = atom_hids.reshape(bs, n, -1) * node_mask.reshape(bs, n, 1)

        # atom_pred = self.node_pred_mlp(atom_hids).reshape(
        #      bs, n, -1) * node_mask.reshape(bs, n, 1)
        # edge_pred = torch.cat([self.edge_exist_mlp(
        #      edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)  # [N_edge, ch]

        # edge_pred = edge_pred + 0.5 * sample_symmetric_edge_feature_noise(bs, edge_pred.shape[1], edge_pred.shape[-1], edge_mask)
        # atom_pred = atom_pred + 0.5 * sample_gaussian_with_mask(size=(atom_pred.shape[0], atom_pred.shape[1], atom_pred.shape[2]), device=node_mask.device,node_mask=node_mask)

    # convert sparse edge_pred to dense form
        edge_final = torch.zeros_like(data.E.float()).reshape(
            bs * n * n, -1)  # [B*N*N, ch]

        edge_final = utils.to_dense_edge_attr(
            edge_index, edge_pred, edge_final, bs, n)

    # post-processing
        pos = pos * node_mask.reshape(-1, 1)
    # pos = (pos - pos_init) * node_mask.reshape(-1, 1)

        if torch.any(torch.isnan(pos)):
           print('Warning: detected nan, resetting output to zero.')
           pos = torch.zeros_like(pos)

        pos = pos.reshape(bs, n, -1)
        # pos = pos + 0.5 * sample_center_gravity_zero_gaussian_with_mask(size=(pos.shape[0],pos.shape[1], 3), device=node_mask.device,
        #                                                                 node_mask=node_mask)
        pos = my_remove_mean_with_mask(pos, node_mask.reshape(bs, n, 1))

    # run Miformer Blocks
        new_E = self.mlp_in_E(edge_final)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
    # print('new_E.shape', new_E.shape)
        features = utils.PlaceHolder(X=atom_hids.reshape(bs, n, -1), E=new_E, y=self.mlp_in_y(data.y), charges=None,
                                     pos=self.mlp_in_pos(pos, node_mask), node_mask=node_mask).mask()

        for layer in self.tf_layers:
            features = layer(features)

        X = self.mlp_out_X(features.X)
        E = self.mlp_out_E(features.E)
        # y = self.mlp_out_y(features.y)
        pos = self.mlp_out_pos(features.pos, node_mask)

        if torch.any(torch.isnan(pos)):
           print('Warning: detected nan, resetting output to zero.')
           pos = torch.zeros_like(pos)

        # pos = pos + 0.5 * sample_center_gravity_zero_gaussian_with_mask(size=(pos.shape[0],pos.shape[1], 3), device=node_mask.device,
        #                                                                 node_mask=node_mask)
        pos = my_remove_mean_with_mask(pos, node_mask.reshape(bs, n, 1))

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        # y = y + y_to_out
        y = y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        final_X = X[..., :self.out_dim_X]
        charges = X[..., self.out_dim_X:]

        out = utils.PlaceHolder(
            pos=pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask).mask()
        return out
    
    
def my_remove_mean_with_mask(x, node_mask):
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask.reshape(size[0], size[1], -1)
    return x_masked


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = my_remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_symmetric_edge_feature_noise(n_samples, n_nodes, edge_ch, edge_mask):
    """sample symmetric normal noise for edge feature."""
    z_edge = torch.randn((n_samples, edge_ch, n_nodes,
                         n_nodes), device=edge_mask.device)
    z_edge = torch.tril(z_edge, -1)
    z_edge = z_edge + z_edge.transpose(-1, -2)
    z_edge = z_edge.permute(0, 2, 3, 1) * \
        edge_mask.reshape(n_samples, n_nodes, n_nodes, 1)
    return z_edge





    #    atom_hids = torch.cat(atom_hids, dim=-1)
    #    edge_hids = torch.cat(edge_hids, dim=-1)
    #    atom_pred = self.node_pred_mlp(atom_hids).reshape(
    #         bs, n, -1) * node_mask.reshape(bs, n, 1)
    #    edge_pred = torch.cat([self.edge_exist_mlp(
    #         edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)  # [N_edge, ch]

    # # convert sparse edge_pred to dense form
    # #edge_final = to_dense_adj(
    #     #edge_index=edge_index, batch=data.batch, edge_attr=edge_pred, max_num_nodes=n)
    #    edge_final = torch.zeros_like(data.E.float()).reshape(
    #         bs * n * n, -1)  # [B*N*N, ch]
    #    edge_final = utils.to_dense_edge_attr(
    #         edge_index, edge_pred, edge_final, bs, n)
    #    edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))

    # # post-processing
    #    pos = pos * node_mask.reshape(-1, 1)
    # #pos = (pos - pos_init) * node_mask.reshape(-1, 1)

    #    if torch.any(torch.isnan(pos)):
    #        print('Warning: detected nan, resetting output to zero.')
    #        pos = torch.zeros_like(pos)

    #    pos = pos.reshape(bs, n, -1)
    #    pos = my_remove_mean_with_mask(pos, node_mask.reshape(bs, n, 1))

    # print('pos.shape', pos.shape)
    # print('node_mask.shape', node_mask.shape)
    # print('edge_final.shape', edge_final.shape)
    # print('atom_pred.shape', atom_pred.shape)

    # run Miformer Blocks
    #    new_E = self.mlp_in_E(edge_final)
    #    new_E = (new_E + new_E.transpose(1, 2)) / 2
    # #print('new_E.shape', new_E.shape)
    #    features = utils.PlaceHolder(X=self.mlp_in_X(atom_pred), E=new_E, y=self.mlp_in_y(data.y), charges=None,
    #                                 pos=self.mlp_in_pos(pos, node_mask), node_mask=node_mask).mask()




    #    # atom_hids = torch.cat(atom_hids, dim=-1)
        # edge_hids = torch.cat(edge_hids, dim=-1)
        # atom_pred = self.node_pred_mlp(atom_hids).reshape(
        #         bs, n, -1) * node_mask.reshape(bs, n, 1)
        # edge_pred = torch.cat([self.edge_exist_mlp(
        #         edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)  # [N_edge, ch]
        # edge_final = torch.zeros_like(data.E.float()).reshape(
        #         bs * n * n, -1)  # [B*N*N, ch]
        # edge_final = utils.to_dense_edge_attr(
        #         edge_index, edge_pred, edge_final, bs, n)
        # edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))