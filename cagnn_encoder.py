import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import AcrE, Cross_Att
import sys

CUDA = torch.cuda.is_available()  # checking cuda availability


class Latent_Learning(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_CAGNN, alpha, nheads_CAGNN):
        super().__init__()
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim = entity_out_dim
        self.nheads_CAGNN = nheads_CAGNN

        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim = relation_out_dim

        # Update_relation
        self.head_rel_attn = [Cross_Att(self.entity_in_dim,
                                        self.entity_out_dim,
                                        drop_CAGNN, alpha,
                                        self.num_nodes,
                                        self.num_relation,
                                        concat=True)
                              for _ in range(self.nheads_CAGNN)]
        self.tail_rel_attn = [Cross_Att(self.entity_in_dim,
                                        self.entity_out_dim,
                                        drop_CAGNN, alpha,
                                        self.num_nodes,
                                        self.num_relation,
                                        concat=True)
                              for _ in range(self.nheads_CAGNN)]

        # Update_entity
        self.head_ent_attn = [Cross_Att(self.entity_in_dim,
                                        self.entity_out_dim,
                                        drop_CAGNN, alpha,
                                        self.num_relation,
                                        self.num_nodes,
                                        concat=True)
                              for _ in range(self.nheads_CAGNN)]
        self.tail_ent_attn = [Cross_Att(self.entity_in_dim,
                                        self.entity_out_dim,
                                        drop_CAGNN, alpha,
                                        self.num_relation,
                                        self.num_nodes,
                                        concat=True)
                              for _ in range(self.nheads_CAGNN)]

        self.drop_CAGNN = drop_CAGNN
        self.alpha = alpha  # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim * self.nheads_CAGNN))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim * self.nheads_CAGNN))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)
        self.w_rel = nn.Parameter(torch.zeros(size=(self.relation_dim + nheads_CAGNN * self.entity_out_dim,
                                                    nheads_CAGNN * self.entity_out_dim)))
        nn.init.xavier_uniform_(self.w_rel.data, gain=1.414)
        self.W_rel = nn.Parameter(torch.zeros(size=(self.relation_dim, nheads_CAGNN * self.entity_out_dim)))
        nn.init.xavier_uniform_(self.W_rel.data, gain=1.414)
        self.w_ent = nn.Parameter(torch.zeros(size=(self.relation_dim + nheads_CAGNN * self.entity_out_dim,
                                                    nheads_CAGNN * self.entity_out_dim)))
        nn.init.xavier_uniform_(self.w_ent.data, gain=1.414)

    def update_relation(self, edge_list, edge_type):
        """
        head_entity  ->  rel  <- tail_entity
        """
        head_ent_embed = self.entity_embeddings[edge_list[0, :]]
        tail_ent_embed = self.entity_embeddings[edge_list[1, :]]
        # rel_embed = self.relation_embeddings[edge_type]
        rel_head = self.relation_embeddings[edge_type + self.num_relation // 2]
        rel_tail = self.relation_embeddings[edge_type]

        head_rel = torch.cat([att_head(edge_list[0, :], head_ent_embed, edge_type, rel_head)
                              for att_head in self.head_rel_attn], dim=1)
        tail_rel = torch.cat([att_tail(edge_list[1, :], tail_ent_embed, edge_type, rel_tail)
                              for att_tail in self.tail_rel_attn], dim=1)
        rel_rep = head_rel + tail_rel
        rel_final = torch.cat((rel_rep, self.relation_embeddings), dim=1)
        rel_final = rel_final.mm(self.w_rel)
        return rel_final

    def update_entity(self, edge_list, edge_type):
        """
        1.
        head_entity  <- rel
        tail_entity  <- rel
        2.
        head_entity  <- rel  <- tail_entity
        """
        head_ent_embed = self.entity_embeddings[edge_list[0, :]]
        tail_ent_embed = self.entity_embeddings[edge_list[1, :]]
        # rel_embed = self.relation_embeddings[edge_type]
        rel_head = self.relation_embeddings[edge_type + self.num_relation // 2]
        rel_tail = self.relation_embeddings[edge_type]

        head_ent = torch.cat([att_head(edge_type, rel_head, edge_list[0, :], head_ent_embed)
                              for att_head in self.head_ent_attn], dim=1)
        tail_ent = torch.cat([att_tail(edge_type, rel_tail, edge_list[1, :], tail_ent_embed)
                              for att_tail in self.tail_ent_attn], dim=1)
        ent_rep = head_ent + tail_ent

        ent_final = torch.cat((ent_rep, self.entity_embeddings), dim=1)
        ent_final = ent_final.mm(self.w_ent)
        # self.final_entity_embeddings.data = ent_rep.data
        return ent_final

    def forward(self, Corpus_, adj, batch_inputs, update_rel=True):
        """
        Args:
             adj: (adj_indices, adj_values)   adj_indices: [(e2, e1), ...]  adj_values:[(rel ...)]
             batch_inputs: [e2, r, e1]   size([batch_size, 3])
        """
        # getting edge_list
        edge_list = adj[0]
        edge_type = adj[1]
        if (CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()

        # if update_rel:
        # 使用实体来更新关系的信息
        rel = self.update_relation(edge_list, edge_type)
        #            print(rel.size())
        # sys.exit()
        # rel.shape: [num_rel, rel_out_dim]
        # print("update rel represent")
        # else:
        #     rel = self.relation_embeddings.mm(self.W_rel)
        # print("not update rel represent")

        # 使用关系来更新实体的信息
        ##实体也进行更新试一试结果怎样
        # ent = self.update_entity(edge_list, edge_type)
        ent = self.entity_embeddings.mm(self.W_rel)
        self.final_entity_embeddings.data = ent.data
        self.final_relation_embeddings.data = rel.data
        return ent, rel


class Explict_Learning(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_acre, alpha, alpha_acre, nheads_CAGNN, conv_out_channels,
                 atrous1_kernel_0, atrous1_dilation_0, atrous2_kernel_0, atrous2_dilation_0):
        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim = entity_out_dim
        self.nheads_CAGNN = nheads_CAGNN

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim = relation_out_dim

        self.drop_acre = drop_acre
        self.alpha = alpha  # For leaky relu
        self.alpha_acre = alpha_acre
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim * self.nheads_CAGNN))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim * self.nheads_CAGNN))

        self.acrE = AcrE(self.entity_out_dim * self.nheads_CAGNN, 3, 1,
                         self.conv_out_channels, self.drop_acre, self.alpha_acre,
                         atrous1_kernel_0, atrous1_dilation_0, atrous2_kernel_0, atrous2_dilation_0)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat(
            (self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
            dim=1)
        out_conv = self.acrE(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat(
            (self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
            dim=1)
        out_conv = self.acrE(conv_input)
        return out_conv
