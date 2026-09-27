import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.size(0)

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.depth
        )
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        return self.dense(output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size):
        super().__init__()
        self.gating_network = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(input_size, output_size) for _ in range(num_experts)]
        )

    def forward(self, x):
        gating_scores = self.gating_network(x)
        gating_weights = F.softmax(gating_scores, dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        weighted_expert_outputs = torch.einsum(
            "be,bue->bu", gating_weights, expert_outputs
        )
        return weighted_expert_outputs


class TransformerDecoderLayerWithMoE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, dropout_rate):
        super(TransformerDecoderLayerWithMoE, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.moe = MixtureOfExperts(num_experts, d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        moe_output = self.moe(x)
        x = x + self.dropout2(moe_output)
        x = self.norm2(x)

        return x
