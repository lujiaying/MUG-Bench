from typing import List, Optional, Final, Dict

import torch as th
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from autogluon.multimodal.models import CategoricalMLP, NumericalMLP, MultimodalFusionMLP
from autogluon.multimodal.constants import CATEGORICAL_MLP, NUMERICAL_MLP, FUSION_MLP
from autogluon.tabular.models.tabular_nn.utils.nn_architecture_utils import get_embed_sizes


ALL_ACT_LAYERS: Final[Dict] = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

FUSION_HIDDEN_SIZE: Final[int] = 128


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


class GAT(nn.Module):
    def __init__(self,
                 in_feats: int,
                 out_feats: int,
                 num_heads: List[int]=[3,1],
                 feat_dropout_prob: float=0.1,
                 residual: bool=True):
        super().__init__()
        self.gnn1 = GATConv(in_feats, in_feats, num_heads=num_heads[0], feat_drop=feat_dropout_prob, residual=residual)
        self.gnn2 = GATConv(in_feats*num_heads[0], out_feats, num_heads=num_heads[1], feat_drop=feat_dropout_prob, residual=residual)

    def forward(self, node_feats: th.Tensor, g: dgl.DGLGraph):
        # current version batch contains the whole graph
        X = F.elu(self.gnn1(g, node_feats))
        X = self.gnn2(g, X.view(X.shape[0], -1))
        X = X.view(X.shape[0], -1)
        return X


class Unit(nn.Module):
    """
    One MLP layer. It orders the operations as: norm -> fc -> act_fn -> dropout
    """

    def __init__(
        self,
        normalization: str,
        in_features: int,
        out_features: int,
        activation: str,
        dropout_prob: float,
    ):
        """
        Parameters
        ----------
        normalization
            Name of activation function.
        in_features
            Dimension of input features.
        out_features
            Dimension of output features.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        """
        super().__init__()
        if normalization == "layer_norm":
            self.norm = nn.LayerNorm(in_features)
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm1d(in_features)
        else:
            raise ValueError(f"unknown normalization: {normalization}")
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # pre normalization
        x = self.norm(x)
        x = self.fc(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP). If the hidden or output feature dimension is
    not provided, we assign it the input feature dimension.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        num_layers: Optional[int] = 1,
        activation: Optional[str] = "leaky_relu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
    ):
        """
        Parameters
        ----------
        in_features
            Dimension of input features.
        hidden_features
            Dimension of hidden features.
        out_features
            Dimension of output features.
        num_layers
            Number of layers.
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        layers = []
        for _ in range(num_layers):
            per_unit = Unit(
                normalization=normalization,
                in_features=in_features,
                out_features=hidden_features,
                activation=activation,
                dropout_prob=dropout_prob,
            )
            in_features = hidden_features
            layers.append(per_unit)
        if out_features != hidden_features:
            self.fc_out = nn.Linear(hidden_features, out_features)
        else:
            self.fc_out = None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        if self.fc_out is not None:
            return self.fc_out(x)
        else:
            return x


class MultiplexGNN(nn.Module):
    def __init__(self,
                 num_categs_per_feature: List[int],
                 numerical_dims: int,
                 text_feats: int,
                 image_feats: int,
                 num_classes: int,
                 gnn_out_feats: int = FUSION_HIDDEN_SIZE,
                 dropout_prob: float = 0.1,
                 ):
        super().__init__()
        # tab_encoder receive both categorical and numerical preproessed features
        self.tab_encoder = TabEncoder(num_categs_per_feature, numerical_dims)
        self.tab_gnn = GAT(self.tab_encoder.out_feats, gnn_out_feats)
        # text modality
        self.txt_encoder = nn.Linear(text_feats, gnn_out_feats)
        self.txt_gnn = GAT(gnn_out_feats, gnn_out_feats)
        # image modality
        self.img_encoder = nn.Linear(image_feats, gnn_out_feats)
        self.img_gnn = GAT(gnn_out_feats, gnn_out_feats)
        # fusion
        self.attn_fc = nn.Sequential(
            nn.Linear(gnn_out_feats, gnn_out_feats//2),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(gnn_out_feats//2, 1, bias=False),
            )
        self.fusion_mlp = MLP(gnn_out_feats, hidden_features=gnn_out_feats//2, 
                              out_features=num_classes, num_layers=1, dropout_prob=dropout_prob)

    def cal_attn(self, fused_X: th.Tensor) -> th.Tensor:
        # fused_X: (*, H)
        scores = self.attn_fc(fused_X)  # (*, 1)
        attns = F.softmax(scores, dim=0)  # (*, 1)
        return attns

    def forward(self, 
                data_batch: Dict[str, th.Tensor],
                tab_g: dgl.DGLGraph,
                txt_g: dgl.DGLGraph,
                img_g: dgl.DGLGraph,
                mask: th.BoolTensor,
                ) -> th.Tensor:
        # data_batch contains key
        # vector: numerical feats
        # embed: categorical feats
        # text: text feats
        # image: image feats
        tab_embs = self.tab_encoder(data_batch)
        tab_embs = self.tab_gnn(tab_embs, tab_g)  # (N_nodes, H)
        txt_embs = self.txt_encoder(data_batch['text'])
        txt_embs = self.txt_gnn(txt_embs, txt_g)
        img_embs = self.img_encoder(data_batch['image'])
        img_embs = self.img_gnn(img_embs, img_g)
        # global attention version
        # cal attention score
        fused_pooled_embs = th.cat([
            th.mean(tab_embs, dim=0, keepdim=True), 
            th.mean(txt_embs, dim=0, keepdim=True),
            th.mean(img_embs, dim=0, keepdim=True),
            ], dim=0)   # (n_modality, H)
        attns = self.cal_attn(fused_pooled_embs)   # (n_modality, 1)
        fused_embs = th.stack([tab_embs, txt_embs, img_embs], dim=-1)  # (N_nodes, H, n_modality)
        fused_embs = th.matmul(fused_embs, attns.squeeze(1))   # (N_nodes, H)
        logits = self.fusion_mlp(fused_embs)  # (N_nodes, n_classes)
        return logits
