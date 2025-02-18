import torch
import torch.nn as nn
from torch.nn import init
import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        new_norm = self.mlp(norm)                              # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """ Note: There is a relatively similar layer implemented by NVIDIA:
            https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
            It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.normalized_shape = (1,)                   # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(node_mask, dim=1, keepdim=True)      # bs, 1, 1
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}'.format(**self.__dict__)


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class EtoX(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """ E: bs, n, n, de"""
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class SetNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, x_mask):
        bs, n, d = x.shape
        divide = torch.sum(x_mask, dim=1, keepdim=True) * d      # bs
        means = torch.sum(x * x_mask, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((x - means) ** 2 * x_mask, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (x - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * x_mask
        return out


class GraphNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, E, emask1, emask2):
        bs, n, _, d = E.shape
        divide = torch.sum(emask1 * emask2, dim=[1, 2], keepdim=True) * d      # bs
        means = torch.sum(E * emask1 * emask2, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((E - means) ** 2 * emask1 * emask2, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (E - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * emask1 * emask2
        return out
    
    
class Trans_Layer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int,
                 heads: int = 1, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Trans_Layer, self).__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_query = Linear(in_channels, heads * out_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor = None
                ) -> Tensor:
        """"""

        H, C = self.heads, self.out_channels

        x_feat = x
        query = self.lin_query(x_feat).view(-1, H, C)
        key = self.lin_key(x_feat).view(-1, H, C)
        value = self.lin_value(x_feat).view(-1, H, C)
        print('edge_attr', edge_attr.shape)
        print('edge_index', edge_index.shape)
        print('query', query.shape)
        print('key', key.shape)
        print('value', value.shape)
        print('x_feat', x_feat.shape)
        
        temp = torch.stack([query, key, value], dim=0)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(
            edge_index, x=temp, edge_attr=edge_attr, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(self, x_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        # Unpack query, key, and value from the stacked tensor
        query_i, key_j, value_j = x_j[0], x_j[1], x_j[2]

        edge_attn = self.lin_edge0(
            edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / \
            math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * \
            torch.tanh(self.lin_edge1(edge_attr).view(-1,
                       self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg
    '''
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(
            edge_attr).view(-1, self.heads, self.out_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / \
            math.sqrt(self.out_channels)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * \
            torch.tanh(self.lin_edge1(edge_attr).view(-1,
                       self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg
'''
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class CondCoorsNorm(nn.Module):
    def __init__(self, time_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 1)
        )

    def forward(self, coors, time_emb):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        scale = self.time_emb(time_emb)
        return normed_coors * scale


class TransMixLayer(MessagePassing):
    """The version for involving the edge feature. Multiply Msg. Without FFN and norm.
    Extra attention heads from adjacency matrix."""

    _alpha: OptTensor

    def __init__(self, x_channels: int, out_channels: int, extra_heads: int = 2,
                 heads: int = 4, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, inf: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.x_channels = x_channels
        self.in_channels = in_channels = x_channels
        self.out_channels = out_channels
        self.heads = heads
        self.extra_heads = extra_heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.sub_heads = sub_heads = heads - extra_heads
        self.sub_channels = sub_channels = (heads * out_channels) // sub_heads
        self.set_inf = inf

        self.lin_key = Linear(in_channels, sub_heads * sub_channels, bias=bias)
        self.lin_query = Linear(
            in_channels, sub_heads * sub_channels, bias=bias)
        self.lin_value = Linear(in_channels, heads * out_channels, bias=bias)

        self.lin_edge0 = Linear(edge_dim, sub_heads * sub_channels, bias=False)
        self.lin_edge1 = Linear(edge_dim, heads * out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge0.reset_parameters()
        self.lin_edge1.reset_parameters()

    def forward(self, x: OptTensor,
                edge_index: Adj,
                edge_attr: OptTensor,
                extra_heads: OptTensor
                ) -> Tensor:
        """"""
        H, E, C, = self.heads, self.sub_heads, self.out_channels

        # expand the extra heads
        cur_extra_heads = extra_heads.size(-1)
        if cur_extra_heads != self.extra_heads:
            n_expand = self.extra_heads // cur_extra_heads
            extra_heads = extra_heads.unsqueeze(-1).repeat(1, 1, n_expand)
            extra_heads = extra_heads.reshape(-1, self.extra_heads)

        x_feat = x
        query = self.lin_query(x_feat).reshape(-1, E, self.sub_channels)
        key = self.lin_key(x_feat).reshape(-1, E, self.sub_channels)
        value = self.lin_value(x_feat).reshape(-1, H, C)


        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out_x = self.propagate(edge_index, query=query, key=key, value=value, edge_attr=edge_attr,
                               extra_heads=extra_heads, size=None)

        out_x = out_x.view(-1, self.heads * self.out_channels)

        return out_x

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor,
                extra_heads: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tuple[Tensor, Tensor]:

        edge_attn = self.lin_edge0(
            edge_attr).view(-1, self.sub_heads, self.sub_channels)
        edge_attn = torch.tanh(edge_attn)
        alpha = (query_i * key_j * edge_attn).sum(dim=-1) / \
            math.sqrt(self.out_channels)

        # set 0 to -inf/1e-10 in extra_heads
        if self.set_inf:
            extra_inf_heads = extra_heads.clone()
            # extra_inf_heads[extra_inf_heads==0.] = -float('inf')
            extra_inf_heads[extra_inf_heads == 0.] = -1e10
            alpha = torch.cat([extra_inf_heads, alpha], dim=-1)
        else:
            alpha = torch.cat([extra_heads, alpha], dim=-1)

        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # node feature message
        msg = value_j
        msg = msg * \
            torch.tanh(self.lin_edge1(edge_attr).view(-1,
                       self.heads, self.out_channels))
        msg = msg * alpha.view(-1, self.heads, 1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features"""

    def __init__(self, K, *args, **kwargs):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, *args, **kwargs):
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)

class CondGaussianLayer(nn.Module):
    """Gaussian basis function layer for 3D distance features, with time embedding condition"""

    def __init__(self, K, time_dim):
        super().__init__()
        self.K = K - 1
        self.means = nn.Embedding(1, self.K)
        self.stds = nn.Embedding(1, self.K)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2)
        )
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x, time_emb=None):
        if time_emb is not None:
            scale, shift = self.time_mlp(time_emb).chunk(2, dim=1)
            x = x * (scale + 1) + shift
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return torch.cat([x, gaussian(x, mean, std).type_as(self.means.weight)], dim=-1)
