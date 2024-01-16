# -*- coding: utf-8 -*-
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import GATConv, SAGEConv
from dgl.nn.pytorch import GraphConv as GCNConv
import dgl.function as fn
from torch.nn import init

class MixHopConv(nn.Module):
    r"""
    Parameters
    ----------
    in_dim : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_dim : int
        Output feature size for each power.
    p: list
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    dropout: float, optional
        Dropout rate on node features. Defaults: ``0``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    batchnorm: bool, optional
        If True, use batch normalization. Defaults: ``False``.
    """
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))
        
        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            # Normalization coefficient of node features
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):
                
                # graph convolution operation on jth layer
                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)
                # Node features are normalized
                feats = feats * norm
                graph.ndata['h'] = feats
                # Sum and aggregate received messages
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                # Updated features
                feats = graph.ndata.pop('h')
                feats = feats * norm
            
            final = torch.cat(outputs, dim=1)
            
            if self.batchnorm:
                final = self.bn(final)
            
            if self.activation is not None:
                final = self.activation(final)
            
            final = self.dropout(final)

            return final

# Semantic Attention
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z): 
        # attention weight
        w = self.project(z)
        w = w.mean(1)
        beta = torch.softmax(w, dim=0)
        # Extended beta has the same dimensions as z
        beta = beta.unsqueeze(1).expand(-1, z.shape[1], -1)
        return beta

class HeteroLinear(nn.Module):
    """Apply linear transformations on heterogeneous inputs.

    Parameters
    ----------
    in_size : dict[key, int]
        Input feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
    out_size : int
        Output feature size.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.
    """
    def __init__(self, in_size, out_size, bias=True):
        super(HeteroLinear, self).__init__()

        self.linears = nn.ModuleDict()
        for typ, typ_in_size in in_size.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, out_size, bias=bias)

    def forward(self, feat):
        """Forward function

        Parameters
        ----------
        feat : dict[key, Tensor]
            Heterogeneous input features. It maps keys to features.

        Returns
        -------
        dict[key, Tensor]
            Transformed features.
        """
        out_feat = dict()
        for typ, typ_feat in feat.items():
            out_feat[typ] = self.linears[str(typ)](typ_feat)

        return out_feat

class ieHGCN(nn.Module):
    r"""
    Parameters
    ----------
    num_layers: int
        the number of layers
    in_dim: int
        the input dimension
    hidden_dim: int
        the hidden dimension
    out_dim: int
        the output dimension
    attn_dim: int
        the dimension of attention vector
    ntypes: list
        the node type of a heterogeneous graph
    etypes: list
        the edge type of a heterogeneous graph
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, attn_dim, ntypes, etypes, bias, batchnorm, dropout):
        super(ieHGCN, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.hgcn_layers = nn.ModuleList()
        
        # define ieHGCNConv num_layers
        self.hgcn_layers.append(
                ieHGCNConv(
                    in_dim,
                    hidden_dim,
                    attn_dim,
                    ntypes,
                    etypes,
                    self.activation,
                    bias,
                    batchnorm,
                    dropout
                ))
        for i in range(1, num_layers - 1):
            self.hgcn_layers.append(
                ieHGCNConv(
                    hidden_dim,
                    hidden_dim,
                    attn_dim,
                    ntypes,
                    etypes,
                    self.activation,
                    bias,
                    batchnorm,
                    dropout
                )
            )
        
        self.hgcn_layers.append(
            ieHGCNConv(
                hidden_dim,
                out_dim,
                attn_dim,
                ntypes,
                etypes,
                None,
                False,
                False,
                0.0
            )
        )

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCN.
        
        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        # hg is a dgl heterogeneous graph
        if hasattr(hg, "ntypes"):
            for l in range(self.num_layers):
                h_dict = self.hgcn_layers[l](hg, h_dict)
        else:
            for layer, block in zip(self.hgcn_layers, hg):
                h_dict = layer(block, h_dict)
        return h_dict

class ieHGCNConv(nn.Module):
    r"""
    The ieHGCN convolution layer.

    Parameters
    ----------
    in_size: int
        the input dimension
    out_size: int
        the output dimension
    attn_size: int
        the dimension of attention vector
    ntypes: list
        the node type list of a heterogeneous graph
    etypes: list
        the edge type list of a heterogeneous graph
    activation: str
        the activation function
    bias: boolean
        whether we need bias vector
    batchnorm: boolean
        whether we need batchnorm
    dropout: float
        the drop out rate
    """
    def __init__(self, in_size, out_size, attn_size, ntypes, etypes, activation = F.elu, 
                 bias = False, batchnorm = False, dropout = 0.0):
        super(ieHGCNConv, self).__init__()
        self.bias = bias
        self.batchnorm = batchnorm
        self.dropout = dropout
        node_size = {}
        for ntype in ntypes:
            node_size[ntype] = in_size
        # define HeteroLinear layer
        self.W_self = HeteroLinear(node_size, out_size)
        
        # define SemanticAttention layer
        self.semantic_attention = SemanticAttention(attn_size)

        self.in_size = in_size
        self.out_size = out_size
        self.attn_size = attn_size
        # define gcn layer
        mods = {
            etype: dglnn.GraphConv(in_size, out_size, norm = 'right', 
                                   weight = True, bias = True, allow_zero_in_degree = True)
            for etype in etypes
            }
        self.mods = nn.ModuleDict(mods)
        
        self.linear_k = nn.ModuleDict({ntype: nn.Linear(out_size, attn_size) for ntype in ntypes})
        self.linear_r = nn.ModuleDict({ntype: nn.Linear(in_size, out_size) for ntype in ntypes})
        
        self.activation = activation
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_size)
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_size))
            nn.init.zeros_(self.h_bias)      
        self.dropout = nn.Dropout(dropout)

    def forward(self, hg, h_dict):
        """
        The forward part of the ieHGCNConv.
        
        Parameters
        ----------
        hg : object or list[block]
            the dgl heterogeneous graph or the list of blocks
        h_dict: dict
            the feature dict of different node types
            
        Returns
        -------
        dict
            The embeddings after final aggregation.
        """
        # outputs: the results of gcn
        outputs = {ntype: [] for ntype in hg.dsttypes}
        # key_nb: the results of outputs processing
        key_nb = {ntype: [] for ntype in hg.dsttypes}
        if hg.is_block:
            src_inputs = h_dict
            dst_inputs = {k: v[:hg.number_of_dst_nodes(k)] for k, v in h_dict.items()}
        # hg is a dgl heterogeneous graph
        else:
            src_inputs = h_dict
            dst_inputs = h_dict
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            # node self gcn
            dst_inputs = self.W_self(dst_inputs)
            key = {}
            attention = {}
            for ntype in hg.dsttypes:
                # outputs append node self
                outputs[ntype].append(dst_inputs[ntype])
                # key_nb append node self
                key[ntype] = self.linear_k[ntype](dst_inputs[ntype])
                key_nb[ntype].append(key[ntype])
            
            # Information transfer for each edge type in heterogeneous graphs
            for srctype, etype, dsttype in hg.canonical_etypes:
                
                # Get the graph of a specific edge type
                rel_graph = hg[srctype, etype, dsttype]
                if srctype not in h_dict:
                    continue
                # use gcn to process
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[srctype], dst_inputs[dsttype])
                )
                
                # if source node and target node are the same
                if(srctype==dsttype):
                    # Average the same nodes
                    outputs[dsttype][0] = (outputs[dsttype][0] + dstdata) / 2
                    dstdata = self.linear_k[dsttype](dstdata)
                    key_nb[dsttype][0] = (key_nb[dsttype][0] + dstdata) / 2
                else:
                    # directly append neighbor node
                    outputs[dsttype].append(dstdata)
                    key_nb[dsttype].append(self.linear_k[dsttype](dstdata))
            
            # get attention coefficient about key_nb
            for ntype in hg.dsttypes:
                key_nb[ntype] = torch.stack((key_nb[ntype]), dim=0)
                attention[ntype] = self.semantic_attention(key_nb[ntype])
            
            # weighted fusion
            rst = {ntype: 0 for ntype in hg.dsttypes}
            for ntype, data in outputs.items():
                if len(data) != 0:
                    for i in range(len(data)):
                        aggregation = torch.mul(data[i], attention[ntype][i])
                        rst[ntype] = aggregation + rst[ntype]

            # residual
            for ntype in hg.dsttypes:
                rst[ntype] = (rst[ntype] + self.linear_r[ntype](h_dict[ntype])) / 2
                
        def _apply(ntype, h):
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            if self.batchnorm:
                h = self.bn(h)
            return self.dropout(h)
            
        return {ntype: _apply(ntype, feat) for ntype, feat in rst.items()}


class ieHGCN_DTI(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, attn_dim, ntypes, etypes, bias, batchnorm, dropout):
        super(ieHGCN_DTI, self).__init__()
        self.sum_layers = ieHGCN(num_layers, in_dim, hidden_dim, out_dim, attn_dim, ntypes, etypes, bias, batchnorm, dropout)

    def forward(self, s_g, s_h):
        # drug and protein feature
        h = self.sum_layers(s_g, s_h)
        return h['drug'], h['protein']

# GCN
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, out_size)
        self.dropout = dropout

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.conv2(g, x)
        return x

# GAT
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(GAT, self).__init__()
        self.num_heads = 4
        self.conv1 = GATConv(in_size, hid_size, self.num_heads)
        self.conv2 = GATConv(hid_size*self.num_heads, out_size, self.num_heads)
        self.dropout = dropout
    def forward(self, g, x):
        x = self.conv1(g, x).flatten(1)
        x = F.elu(x)
        x = F.dropout(x, self.dropout)
        x = self.conv2(g, x).mean(1)
        return x

# SAGE
# aggregator type: mean/pool/lstm/gcn
class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_size, hid_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hid_size, out_size, aggregator_type='mean')
        self.dropout = dropout

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.conv2(g, x)
        return x

# MixHop
class MixHop(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout):
        super(MixHop, self).__init__()
        self.conv1 = MixHopConv(in_size, hid_size)
        self.conv2 = MixHopConv(3 * hid_size, out_size)
        self.dropout = dropout
        self.fusion = nn.Sequential(
            nn.Linear(3 * out_size, out_size),
            nn.ReLU(True),
            nn.LayerNorm(out_size),
            nn.Linear(out_size, out_size))
        
    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.conv2(g, x)
        x = self.fusion(x)
        return x

# predictor mlp
class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 128, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(128, 16, bias=False),
            nn.ELU(),
            nn.Linear(16, 2, bias=False),
            nn.LogSoftmax(dim=1))
            # nn.Sigmoid())
    def forward(self, x):
        output = self.MLP(x)
        return output


class ADEM(nn.Module):
    def __init__(self, args):
        super(ADEM, self).__init__()
        self.args = args
        self.ieHGCN_DTI = ieHGCN_DTI(args.num_layers, args.in_size, args.hidden_size, args.out_size, args.attn_size, args.ntypes, args.etypes, args.bias, args.batchnorm, args.dropout)
        
        # dpp_graph node feature size, actually it's 2*args.dpg_size
        self.fusion = nn.Sequential(
            nn.Linear(args.out_size, args.dpg_size),
            nn.ReLU(True),
            nn.LayerNorm(args.dpg_size),
            nn.Linear(args.dpg_size, args.dpg_size))
        self.lin = nn.Linear(2 * args.dpg_size, args.out_size)
        
        # the choice to process dpp_graph
        if args.gnn_type == 'mixhop':
            self.gnn = MixHop(2 * args.dpg_size, args.hidden_size, args.hidden_size, args.dropout)
        elif args.gnn_type == 'gcn':
            self.gnn = GCN(2 * args.dpg_size, args.hidden_size, args.hidden_size, args.dropout)
        elif args.gnn_type == 'gat':
            self.gnn = GAT(2 * args.dpg_size, args.hidden_size, args.hidden_size, args.dropout)
        elif args.gnn_type == 'sage': 
            self.gnn = SAGE(2 * args.dpg_size, args.hidden_size, args.hidden_size, args.dropout)
            
        self.MLP = MLP(args.hidden_size)
        
    def forward(self, graph, h, dateset_index, data, edge, iftrain=True, d=None, p=None):
        
        if iftrain:
            # get the drug and protein feature
            drug_hgnn, protein_hgnn = self.ieHGCN_DTI(graph, h)
            d = self.fusion(drug_hgnn)
            p = self.fusion(protein_hgnn)
        # use dpp_graph
        if self.args.predictor == 'dpg':
            # constructure dpp_graph
            dp_graph, dp_feature = constructure_graph(data, edge, d, p)
            dp_graph = dp_graph.to(self.args.device)
            dp_feature = dp_feature.to(self.args.device)
            feature = self.gnn(dp_graph, dp_feature)
        # direct prediction
        elif self.args.predictor == 'lin':
            feature = torch.cat((d[data[:, :1]], p[data[:, 1:2]]), dim=2).squeeze(1)
            feature = self.lin(feature)
        
        pred = self.MLP(feature[dateset_index])

        if iftrain:
            return pred, d, p
        return pred

def init(i):
    if isinstance(i, nn.Linear):
        nn.init.xavier_uniform_(i.weight)