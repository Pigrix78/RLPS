import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        bound = 1.0 / math.sqrt(out_features)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, node_features, adjacency):
        output = adjacency @ (node_features @ self.weight) + self.bias
        return F.normalize(output, p=2, dim=1)


class PointWiseExtractor(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        dimensions = (input_channels, 24, 32, 48, 72, 104)
        layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                (
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, points):
        # [batch, locations, channels] -> [batch, channels, locations, 1]
        return self.layers(points.transpose(1, 2).unsqueeze(-1)).squeeze(-1)


class ExternalAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, memory_units: int = 40, dropout: float = 0.05):
        super().__init__()
        if dimension % heads:
            raise ValueError("The feature dimension must be divisible by the number of heads.")
        self.heads = heads
        self.head_dimension = dimension // heads
        self.memory_units = memory_units
        self.to_heads = nn.Linear(dimension, dimension)
        self.to_memory = nn.Linear(self.head_dimension, memory_units)
        self.from_memory = nn.Linear(memory_units, self.head_dimension)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(dimension, dimension)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        batch, count, dimension = tokens.shape
        values = self.to_heads(tokens).view(batch, count, self.heads, self.head_dimension)
        values = values.permute(0, 2, 1, 3)
        attention = self.to_memory(values).softmax(dim=-2)
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1.0e-9)
        attention = self.attention_dropout(attention)
        output = self.from_memory(attention).permute(0, 2, 1, 3).reshape(batch, count, dimension)
        return self.projection_dropout(self.projection(output))


class FeedForward(nn.Module):
    def __init__(self, dimension: int, dropout: float = 0.05):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        return self.layers(tokens)


def stochastic_depth(tensor, probability: float, training: bool):
    if probability == 0.0 or not training:
        return tensor
    keep = 1.0 - probability
    shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    mask = tensor.new_empty(shape).bernoulli_(keep).div_(keep)
    return tensor * mask


class ContextBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = ExternalAttention(dimension, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension, dropout)
        self.drop_path = dropout

    def forward(self, tokens):
        tokens = tokens + stochastic_depth(self.attention(self.norm1(tokens)), self.drop_path, self.training)
        return tokens + stochastic_depth(self.feed_forward(self.norm2(tokens)), self.drop_path, self.training)


class PSGCN(nn.Module):
    """Point-Set Graph-Regularized Classification Network."""

    def __init__(self, input_channels: int, classes: int):
        super().__init__()
        self.extractor = PointWiseExtractor(input_channels)
        self.context = nn.ModuleList((ContextBlock(104, 4), ContextBlock(104, 8)))
        self.classifier = nn.Linear(104, classes)
        self.graph_hidden = GraphConvolution(104, 24)
        self.graph_output = GraphConvolution(24, classes)

    def forward(self, points, adjacency=None):
        tokens = self.extractor(points).transpose(1, 2)
        for block in self.context:
            tokens = block(tokens)
        probabilities = self.classifier(tokens.mean(dim=1)).softmax(dim=1)

        graph_targets = None
        if adjacency is not None:
            graph_nodes = F.relu(self.graph_hidden(self.classifier.weight, adjacency))
            graph_targets = self.graph_output(graph_nodes, adjacency).softmax(dim=1)
        return probabilities, graph_targets
