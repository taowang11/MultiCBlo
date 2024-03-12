from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from dgllife.model.gnn.gat import GAT
from dgl.nn.pytorch import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import tokens_struct
from torch_geometric.nn import GraphNorm
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class Model(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, num_step_set2set=6,
                 num_layer_set2set=3, rnn_embed_dim=64, blstm_dim=128, blstm_layers=2, fp_2_dim=128, num_heads=4,
                 dropout=0.2, device='cpu'):
        super(Model, self).__init__()
        self.device = device
        self.vocab = tokens_struct()
        if hidden_feats is None:
            hidden_feats = [64, 64]
        self.final_hidden_feats = hidden_feats[-1]
        # self.norm_layer_module = nn.LayerNorm(self.final_hidden_feats).to(device)
        self.gnn = GNNModule(in_feats, hidden_feats, dropout, num_step_set2set, num_layer_set2set)
        self.rnn = RNNModule(self.vocab, rnn_embed_dim, blstm_dim, blstm_layers, self.final_hidden_feats, dropout,
                             bidirectional=True, device=device)
        self.fp_mlp = FPNModule(fp_2_dim, self.final_hidden_feats)
        self.conv = nn.Sequential(nn.Conv2d(12, 12, kernel_size=3), nn.ReLU(),
                                  nn.Dropout(dropout))

        self.mlp1 = nn.Sequential(
            nn.Linear(384, 256),

            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()
        # self.separator=Separator()
        self.Layernorm1 = nn.LayerNorm(384)
        self.Layernorm2 = nn.LayerNorm(384)
        self.Layernorm3 = nn.LayerNorm(384)
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=2, dim_feedforward=64)
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        path_encoder_layer1 = nn.TransformerEncoderLayer(d_model=384, nhead=2, dim_feedforward=64)
        self.path_transformer1 = nn.TransformerEncoder(path_encoder_layer1, num_layers=2)
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=2, dim_feedforward=64)
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        modal3_transformer = nn.TransformerEncoderLayer(d_model=384, nhead=2, dim_feedforward=64)
        self.modal3_transformer = nn.TransformerEncoder(modal3_transformer, num_layers=2)

    def forward(self, padded_smiles_batch, batch, fp_t):
        # get graph input
        batch_size = padded_smiles_batch.size(0)
        smiles_x = self.Layernorm1(self.rnn(padded_smiles_batch, batch.seq_len)).view(batch_size, 1, -1)
        fp_x = self.Layernorm2(self.fp_mlp(fp_t)).view(batch_size, 1, -1)
        graph, loss2 = self.gnn(batch.x, batch.edge_index, batch.batch)
        graph_x = self.Layernorm3(graph).view(batch_size, 1, -1)

        for i in range(1):
            if i == 0:
                out_img = (graph_x)
                out_omic = (fp_x)
                out_modal3 = (smiles_x)

                attention_scores = torch.matmul(out_omic.permute(0, 2, 1), out_modal3)
                attention_weights = F.softmax(attention_scores, dim=-1)

                out_fused1 = torch.matmul(attention_weights, out_modal3.permute(0, 2, 1))

                # 第二步：将 out_fused1 与 modal3_ 融合
                # 注意：这里假设 out_fused1 和 out_modal3 的维度匹配，或者已经通过适当的变换使它们匹配
                attention_scores = torch.matmul(out_img, out_fused1)
                attention_weights = F.softmax(attention_scores, dim=-1)
                out_fused2 = torch.matmul(attention_weights, out_fused1.permute(0, 2, 1))
                # # out = self.mlp1(out_fused2)
            else:
                out_img = (graph_x)
                out_omic = (fp_x)
                out_modal3 = (out_fused2)

                attention_scores = torch.matmul(out_omic.permute(0, 2, 1), out_modal3)
                attention_weights = F.softmax(attention_scores, dim=-1)
                out_fused1 = torch.matmul(attention_weights, out_img.permute(0, 2, 1))

                # 第二步：将 out_fused1 与 modal3_ 融合
                # 注意：这里假设 out_fused1 和 out_modal3 的维度匹配，或者已经通过适当的变换使它们匹配
                attention_scores = torch.matmul(out_modal3, out_fused1)
                attention_weights = F.softmax(attention_scores, dim=-1)
                out_fused2 = torch.matmul(attention_weights, out_fused1.permute(0, 2, 1))


        out = self.mlp1(smiles_x.squeeze(1))

        return out, 0

    def predict(self, smiles, graphs, atom_feats, fp_t):
        return self.sigmoid(self.forward(smiles, graphs, atom_feats, fp_t))


class GNNModule(nn.Module):
    def __init__(self, in_feats=64, hidden_feats=None, dropout=0.2, num_step_set2set=6,
                 num_layer_set2set=3):
        super(GNNModule, self).__init__()
        self.conv = GAT(in_feats, hidden_feats)
        self.readout = Set2Set(input_dim=hidden_feats[-1],
                               n_iters=num_step_set2set,
                               n_layers=num_layer_set2set)
        self.norm = GraphNorm(hidden_feats[-1] * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_feats[-1] * 2, hidden_feats[-1]), nn.ReLU(),
                                nn.Dropout(p=dropout))
        num_features_xd = 84

        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 384)
        )
        self.relu = nn.ReLU()
        self.vq = VectorQuantize(dim=num_features_xd * 10,
                                 codebook_size=4000,
                                 commitment_weight=0.1,
                                 decay=0.9)

    def vector_quantize(self, f, vq_model):
        v_f, indices, v_loss = vq_model(f)

        return v_f, v_loss

    def forward(self, x, edge_index, batch):
        x_g = self.relu(self.conv1(x, edge_index))
        x_g = self.relu(self.conv2(x_g, edge_index))
        # node_v_feat, cmt_loss = self.vector_quantize(x_g.unsqueeze(0), self.vq)
        # node_v_feat = node_v_feat.squeeze(0)
        # node_res_feat = x_g + node_v_feat
        x_g = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        x_g = self.fc_g(x_g)
        return x_g, 0


class RNNModule(nn.Module):
    def __init__(self, vocab, embed_dim, blstm_dim, num_layers, out_dim=2, dropout=0.2, bidirectional=True,
                 device='cpu'):
        super(RNNModule, self).__init__()
        # self.vocab = vocab
        # self.embed_dim = embed_dim
        # self.blstm_dim = blstm_dim
        # self.hidden_size = blstm_dim
        # self.num_layers = num_layers
        # self.out_dim = out_dim
        # self.bidirectional = bidirectional
        # self.device = device
        # self.num_dir = 1
        # if self.bidirectional:
        #     self.num_dir += 1
        # self.embeddings = nn.Embedding(vocab.tokens_length, self.embed_dim, padding_idx=vocab.pad)
        # self.rnn = nn.LSTM(self.embed_dim, self.blstm_dim, num_layers=self.num_layers,
        #                    bidirectional=self.bidirectional, dropout=dropout,
        #                    batch_first=True)
        # self.drop = nn.Dropout(p=dropout)
        # self.fc = nn.Sequential(nn.Linear(2 * self.blstm_dim, self.out_dim), nn.ReLU(), nn.Dropout(p=dropout))
        # # if self.bidirectional:
        # #     # self.norm_layer = nn.LayerNorm(2 * self.blstm_dim).to(device)
        # #     self.fc = nn.Sequential(nn.Linear(2 * self.blstm_dim, self.out_dim), nn.ReLU(),
        # #                             nn.Dropout(p=dropout))
        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)
        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.linear = nn.Sequential(
            nn.Linear(200, 512),
            nn.Linear(512, 384)
        )

    def forward(self, batch, seq_len):
        smi_em = batch.view(-1, 100, 100).float()
        smi_em, _ = self.W_rnn(smi_em)
        smi_em = torch.relu(smi_em)
        sentence_att = self.softmax(torch.tanh(self.fc(smi_em)), 1)
        smi_em = torch.sum(sentence_att.transpose(1, 2) @ smi_em, 1) / 10
        smi_em = self.linear(smi_em)
        return smi_em

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        soft_max_2d = F.softmax(trans_input.contiguous().view(-1, trans_input.size()[-1]), dim=1)
        return soft_max_2d.view(*trans_input.size()).transpose(axis, len(input_size) - 1)


class FPNModule(nn.Module):
    def __init__(self, fp_2_dim, out_feats, dropout=0.2):
        super(FPNModule, self).__init__()
        self.fp_2_dim = fp_2_dim
        self.dropout_fpn = dropout
        self.out_feats = out_feats
        self.fp_dim = 2513
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.out_feats)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smiles):
        fpn_out = self.fc1(smiles)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out


class Separator(nn.Module):
    def __init__(self):
        super(Separator, self).__init__()
        # if args.dataset.startswith('GOOD'):
        #     # GOOD
        #     # if config.model.model_name == 'GIN':
        #     #     self.r_gnn = GINFeatExtractor(config, without_readout=True)
        #     # else:
        #     #     self.r_gnn = vGINFeatExtractor(config, without_readout=True)
        #     emb_d = config.model.dim_hidden
        # else:
        #     self.r_gnn = GNN_node(num_layer=args.layer, emb_dim=args.emb_dim,
        #                           drop_ratio=args.dropout, gnn_type=args.gnn_type)
        # num_features_xd=93

        emb_d = 1152

        self.separator = nn.Sequential(nn.Linear(emb_d, emb_d * 2),
                                       # nn.BatchNorm1d(emb_d * 2),
                                       nn.ReLU(),
                                       nn.Linear(emb_d * 2, emb_d),
                                       nn.Sigmoid())
        # self.args = args
        # self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        # self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.relu = nn.ReLU()

    def forward(self, data):
        # x_g = self.relu(self.conv1(data.x, data.edge_index))
        #
        # x_g = self.relu(self.conv2(x_g, data.edge_index))
        score = self.separator(data)  # [n, d]

        # reg on score

        pos_score_on_node = score.mean(1)  # [n]
        pos_score_on_batch = pos_score_on_node  # [B]
        neg_score_on_batch = 1 - pos_score_on_node  # [B]
        return score, pos_score_on_batch + 1e-8, neg_score_on_batch + 1e-8
