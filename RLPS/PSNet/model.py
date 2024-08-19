import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic(nn.Module):
    def __init__(self, input_channel):
        super(Basic, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 64, [1, 1])
        self.conv2 = nn.Conv2d(64, 64, [1, 1])
        self.conv3 = nn.Conv2d(64, 64, [1, 1])
        self.conv4 = nn.Conv2d(64, 128, [1, 1])
        self.conv5 = nn.Conv2d(128, 256, [1, 1])
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.transpose(2, 1).unsqueeze(-1)
        # print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # x = torch.max(x, 2, keepdim=True)[0]
        x = x.squeeze(dim=-1)  # torch.Size([256, 256, 49])
        # x = torch.sum(x, 2)
        # x = x.view(-1, 2048)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.coef = 4
        self.trans_dims = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.num_heads * self.coef
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x)  # B, N, C
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = self.linear_0(x)

        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PSNet(nn.Module):
    def __init__(self, input_channel, k=2):
        super(PSNet, self).__init__()
        self.basic = Basic(input_channel)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)

        self.AttenMLP = nn.ModuleList([Block(256, num_heads=4) for _ in range(1)])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.basic(x)
        x = x.permute(0, 2, 1)  # torch.Size([256, 49, 256])

        for encoder in self.AttenMLP:
            x = encoder(x) #torch.Size([256, 49, 256])
        x = x.permute(0, 2, 1).view(-1, 256, 7, 7) #torch.Size([256, 256, 7, 7])
        x = self.avgpool(x) #torch.Size([256, 256, 1, 1])
        x = x.view(x.size(0), -1) #torch.Size([256, 256])

        x = self.fc3(x)
        return F.softmax(x, dim=1)