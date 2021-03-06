import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()


class AcrE(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky,
                 atrous1_kernel_0,atrous1_dilation_0,atrous2_kernel_0,atrous2_dilation_0):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len), padding=(0,1))  # kernel size -> 1*input_seq_length(i.e. 2)
        print(atrous1_kernel_0,atrous1_dilation_0,atrous2_kernel_0,atrous2_dilation_0,out_channels)
        # atrous1_kernel_0 = 2
        # atrous1_dilation_0 = 2
        # atrous2_kernel_0 = 3
        # atrous2_dilation_0 = 5
        atrous1_padding_0 = (atrous1_kernel_0+(atrous1_dilation_0-1)*(atrous1_kernel_0-1)-1)
        atrous2_padding_0 = (atrous2_kernel_0+(atrous2_dilation_0-1)*(atrous2_kernel_0-1)-1)
        #卷积核和距离需要奇偶相同
        assert atrous1_padding_0 % 2 == 0
        assert atrous2_padding_0 % 2 == 0
        atrous2_padding_0 = atrous2_padding_0 // 2
        atrous1_padding_0 = atrous1_padding_0 // 2
        self.atrous_conv_layer1 = nn.Conv2d(
            out_channels, out_channels, (atrous1_kernel_0, input_seq_len),
            dilation=(atrous1_dilation_0,1), padding=(atrous1_padding_0,1)
        )
        self.atrous_conv_layer2 = nn.Conv2d(
            out_channels, out_channels, (atrous2_kernel_0, input_seq_len),
            dilation=(atrous2_dilation_0,1), padding=(atrous2_padding_0,1)
        )
        self.out_conv_layer = nn.Conv2d(
            out_channels, out_channels, (1, input_seq_len))
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        ##换层两层普通卷积试一试结果
        # res = conv_input
        # conv_input = self.conv_layer(conv_input).relu()
        # conv_input = self.out_conv_layer(conv_input + res)
        # out_conv = self.dropout(
        #     self.non_linearity(conv_input))
        
        conv_input = self.conv_layer(conv_input)
        res = conv_input
        conv_input = self.atrous_conv_layer1(conv_input)
        conv_input = self.atrous_conv_layer2(conv_input)
        conv_input = self.out_conv_layer(conv_input+res)
        out_conv = self.dropout(
            self.non_linearity(conv_input))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SparseAttnFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, mat_size, E, out_features):
        """
        Args:
            edge: shape=(2, total)
            edge_w: shape=(total, 1)
        """
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([mat_size[0], mat_size[1], out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None
class SparseAttn(nn.Module):
    def forward(self, edge, edge_w, mat_size, E, out_features):
        return SparseAttnFunction.apply(edge, edge_w, mat_size, E, out_features)

class Cross_Att(nn.Module):
    def __init__(self, embed_dim, embed_out_dim, dropout, alpha, num_key, num_query, concat=True):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(
            size=(embed_out_dim, 2 * embed_dim),device='cpu'))#device='cuda'
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, embed_out_dim),device='cpu'))#device='cuda'
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)
        self.trans = nn.Parameter(torch.zeros(size=(embed_out_dim, embed_dim ),device='cpu'))#device='cuda'
        nn.init.xavier_normal_(self.trans.data, gain=1.414)
        self.spareattn = SparseAttn()
        self.num_key = num_key
        self.num_query = num_query
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.embed_out_dim = embed_out_dim
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, key_list, key_embed, query_list, query_embed):
        """
        ent_embed : shape=(num_tripes, embed_dim)
        ent_list : shape=(num_tripes, )
        rel_embed : shape=(num_tripes, embed_dim)
        rel_list : shape=(num_tripes, )
        """
        #edge_h.shape=(embed_dim * 2, num_tripes)

        edge_associated_set = self.trans.mm(key_embed.t())
        # edge_associated_set = key_embed.t()

        edge_h = torch.cat((key_embed, query_embed), dim=1).t()
        #edge_m.shape=(hidden, num_tripes)
        edge_m = self.a.mm(edge_h)


        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        edge_e = torch.exp(powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()

        edge = torch.cat([query_list.unsqueeze(0), key_list.unsqueeze(0)], dim=0)

        e_rowsum = self.spareattn(
            edge, edge_e, (self.num_query, self.num_key), edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E

        #edge_w = (edge_e * edge_m).t()  # eg 8　分子
        #更新实体使用关系的信息
        #更新关系使用实体的信息
        edge_w = (edge_e * edge_associated_set).t()


        # edge_w: E * D
        h_prime = self.spareattn(
            edge, edge_w, (self.num_query, self.num_key), edge_w.shape[0], self.embed_out_dim)

        # h_prime = self.spareattn(
        #     edge, edge_w, (self.num_query, self.num_key), edge_w.shape[0], edge_w.shape[1])

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime
