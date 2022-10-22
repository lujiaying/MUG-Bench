from typing import List, Dict

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv
from autogluon.multimodal.models import CategoricalMLP, NumericalMLP, MultimodalFusionMLP
from autogluon.multimodal.constants import CATEGORICAL_MLP, NUMERICAL_MLP, FUSION_MLP
from autogluon.tabular.models.tabular_nn.utils.nn_architecture_utils import get_embed_sizes


Default_CatMLP_Config = dict(
    out_features=128,
    activation="leaky_relu",
    num_layers=1,
    dropout_prob=0.1,
    normalization="layer_norm",
)

Default_NumMLP_Config = dict(
    hidden_features=128,
    out_features=128,
    activation="leaky_relu",
    num_layers=1,
    dropout_prob=0.1,
    normalization="layer_norm",
)

Default_FusionMLP_Config = dict(
    adapt_in_features="max",
    hidden_features=[128],
    activation="leaky_relu",
    dropout_prob=0.1,
    normalization="layer_norm",
)

Default_GCN_Config = dict(
    in_feats=128,
    out_feats=128,
    norm='both',
)


class tabGNN(nn.Module):
    def __init__(self, 
                 num_categories: List[int],
                 num_numerical_feats: int,
                 num_classes: int,
        ):
        super().__init__()
        # set tabular feature encoder
        global Default_CatMLP_Config, Default_NumMLP_Config, Default_FusionMLP_Config
        self.cat_mlp = CategoricalMLP(CATEGORICAL_MLP, num_categories, **Default_CatMLP_Config)
        self.num_mlp = NumericalMLP(NUMERICAL_MLP, num_numerical_feats, **Default_NumMLP_Config)
        self.tab_encoder = MultimodalFusionMLP(FUSION_MLP, models=[self.cat_mlp, self.num_mlp], num_classes=num_classes, **Default_FusionMLP_Config)
        # set gnn
        global Default_GCN_Config
        self.gcn1 = GraphConv(activation=nn.LeakyReLU(), **Default_GCN_Config)
        self.gcn2 = GraphConv(in_feats=self.gcn1._out_feats, out_feats=num_classes, norm='both')

    def forward(self, batch: dict, g: dgl.DGLGraph):
        """
        Please refer to the forward() function of MultimodalFusionMLP
        https://github.com/awslabs/autogluon/blob/master/multimodal/src/autogluon/multimodal/models/fusion.py
        """
        # current version batch contains the whole graph
        ret = self.tab_encoder(batch)
        node_feats = ret[FUSION_MLP]['features']
        node_feats = self.gcn1(g, node_feats, edge_weight=g.edata['w'])
        node_feats = self.gcn2(g, node_feats, edge_weight=g.edata['w'])
        return node_feats


class TabEncoder(nn.Module):
    def __init__(self, 
                 num_categs_per_feature: List[int] = [],
                 vector_dims: int = 0,
                 ):
        super().__init__()
        params = TabEncoder.get_default_params()
        self._out_feats = params['hidden_size']
        # init input size
        input_size = 0

        # define embedding layer:
        self.has_embed_features = (len(num_categs_per_feature) > 0)
        if self.has_embed_features:
            embed_dims = get_embed_sizes(None, params, num_categs_per_feature)
            self.embed_blocks = nn.ModuleList()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.append(nn.Embedding(num_embeddings=num_categs_per_feature[i],
                                                      embedding_dim=embed_dims[i]))
                input_size += embed_dims[i]

        # update input size from vector in_feats
        self.has_vector_features = (vector_dims > 0)
        if self.has_vector_features:
            input_size += vector_dims
        # activation
        if params['activation'] == 'elu':
            act_fn = nn.ELU()
        elif params['activation'] == 'relu':
            act_fn = nn.ReLU()
        elif params['activation'] == 'tanh':
            act_fn = nn.Tanh()
        else:
            act_fn = nn.Identity()

        # define main_block
        layers = []
        if params['use_batchnorm']:
            layers.append(nn.BatchNorm1d(input_size))
        layers.append(nn.Linear(input_size, params['hidden_size']))
        layers.append(act_fn)
        for _ in range(params['num_layers'] - 1):
            if params['use_batchnorm']:
                layers.append(nn.BatchNorm1d(params['hidden_size']))
            layers.append(nn.Dropout(params['dropout_prob']))
            layers.append(nn.Linear(params['hidden_size'], params['hidden_size']))
            layers.append(act_fn)
        layers.pop(-1)  # remove last act_fn
        self.main_block = nn.Sequential(*layers)

    def forward(self, data_batch: dict):
        input_data = []
        if self.has_vector_features:
            input_data.append(data_batch['vector'])
        if self.has_embed_features:
            embed_data = data_batch['embed']
            for i in range(len(self.embed_blocks)):
                input_data.append(self.embed_blocks[i](embed_data[i]))

        if len(input_data) > 1:
            input_data = th.cat(input_data, dim=1).to(th.get_default_dtype())
        else:
            input_data = input_data[0]
        logits = self.main_block(input_data)
        return logits

    @property
    def out_feats(self):
        return self._out_feats


    @staticmethod
    def get_default_params():
        # params = {'activation': 'relu', 'embedding_size_factor': 1.0, 'embed_exponent': 0.56, 'max_embedding_dim': 100, 'y_range': None, 'y_range_extend': 0.05, 'dropout_prob': 0.1, 'optimizer': 'adam', 'learning_rate': 0.0003, 'weight_decay': 1e-06, 'num_layers': 4, 'hidden_size': 128, 'use_batchnorm': False}
        params = {'activation': 'relu', 'embedding_size_factor': 1.0, 'embed_exponent': 0.56, 'max_embedding_dim': 100, 'y_range': None, 'y_range_extend': 0.05, 'dropout_prob': 0.1, 'optimizer': 'adam', 'learning_rate': 0.0003, 'weight_decay': 1e-06, 'num_layers': 2, 'hidden_size': 128, 'use_batchnorm': False}
        return params


class GCN(nn.ModuleList):
    def __init__(self,
                 in_feats: int,
                 num_classes: int):
        super().__init__()
        #self.gnn1 = GraphConv(in_feats=in_feats, out_feats=num_classes, norm='both')
        #self.gnn2 = GraphConv(in_feats=in_feats, out_feats=num_classes, norm='both')
        num_heads = 3
        dropout_prob = 0.1
        self.gnn1 = GATConv(in_feats, in_feats, num_heads=num_heads, feat_drop=dropout_prob, attn_drop=dropout_prob, residual=True, activation=F.elu)
        self.gnn2 = GATConv(in_feats*num_heads, num_classes, num_heads=1, feat_drop=dropout_prob, attn_drop=dropout_prob, residual=True)

    def forward(self, node_feats: th.Tensor, g: dgl.DGLGraph):
        # current version batch contains the whole graph
        # X = self.gnn1(g, node_feats, edge_weight=g.edata['w'])
        # X = self.gnn2(g, X, edge_weight=g.edata['w'])
        X = self.gnn1(g, node_feats)
        X = self.gnn2(g, X.view(X.shape[0], -1))
        X = X.view(X.shape[0], -1)
        return X
