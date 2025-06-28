import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

def drop_path(x, drop_prob=0., training=False):
    """Drop path regularization technique"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class TokenEmbedding(nn.Module):
    """Embedding layer for tokens/patterns"""
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    """Temporal positional encoding"""
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()

class FlowPositionalEncoding(nn.Module):
    """Positional encoding for flows (OD pairs)"""
    def __init__(self, flow_pos_enc, embed_dim):
        super().__init__()
        self.flow_pos_enc = flow_pos_enc  # Precomputed [num_flows, pos_dim]
        self.linear = nn.Linear(flow_pos_enc.size(-1), embed_dim)
        
    def forward(self, x):
        # x: [batch, time, num_flows, features]
        pos_enc = self.linear(self.flow_pos_enc)  # [num_flows, embed_dim]
        return pos_enc.unsqueeze(0).unsqueeze(0)  # [1, 1, num_flows, embed_dim]

class DataEmbedding(nn.Module):
    """Combined data embedding layer for flow-based model"""
    def __init__(
        self, feature_dim, embed_dim, flow_pos_enc, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        
        # Flow-based positional encoding
        self.spatial_embedding = FlowPositionalEncoding(flow_pos_enc, embed_dim)
        
        if self.add_time_in_day:
            self.minute_size = 1440
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
        if self.add_day_in_week:
            weekday_size = 7
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
        
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        origin_x = x
        # Embed main features
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        
        # Add temporal positional encoding
        x += self.position_encoding(x)
        
        # Add flow-based spatial encoding
        x += self.spatial_embedding(x)
        
        # Add time features if enabled
        if self.add_time_in_day:
            time_idx = (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            x += self.daytime_embedding(time_idx)
        if self.add_day_in_week:
            day_idx = origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3)
            x += self.weekday_embedding(day_idx)
        
        return self.dropout(x)

class DropPath(nn.Module):
    """DropPath regularization module"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class STSelfAttention(nn.Module):
    """Spatio-Temporal Self-Attention for flow-based model"""
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads)
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim

        # Pattern-based attention components
        self.pattern_q_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_k_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])
        self.pattern_v_linears = nn.ModuleList([
            nn.Linear(dim, int(dim * self.geo_ratio)) for _ in range(output_dim)
        ])

        # Geometric attention components
        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)

        # Semantic attention components
        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias)
        self.sem_attn_drop = nn.Dropout(attn_drop)

        # Temporal attention components
        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        # Projection layer
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        B, T, F, D = x.shape  # F is num_flows
          
        # Temporal attention (updated to handle dimensions)
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        t_q = t_q.reshape(B, F, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, F, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, F, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, F, T, int(D * self.t_ratio)).transpose(1, 2)

        # Geometric attention with pattern enhancement
        geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for i in range(self.output_dim):
            pattern_q = self.pattern_q_linears[i](x_patterns[..., i])
            pattern_k = self.pattern_k_linears[i](pattern_keys[..., i])
            pattern_v = self.pattern_v_linears[i](pattern_keys[..., i])
            pattern_attn = (pattern_q @ pattern_k.transpose(-2, -1)) * self.scale
            pattern_attn = pattern_attn.softmax(dim=-1)
            geo_k += pattern_attn @ pattern_v
        geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        geo_q = geo_q.reshape(B, T, F, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_k = geo_k.reshape(B, T, F, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_v = geo_v.reshape(B, T, F, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
        if geo_mask is not None:
            geo_attn = geo_attn.masked_fill(geo_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float('-inf'))
        geo_attn = geo_attn.softmax(dim=-1)
        geo_attn = self.geo_attn_drop(geo_attn)
        geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, F, int(D * self.geo_ratio))

        # Semantic attention
        sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        sem_q = sem_q.reshape(B, T, F, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_k = sem_k.reshape(B, T, F, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_v = sem_v.reshape(B, T, F, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
        if sem_mask is not None:
            sem_attn = sem_attn.masked_fill(sem_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float('-inf'))
        sem_attn = sem_attn.softmax(dim=-1)
        sem_attn = self.sem_attn_drop(sem_attn)
        sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, F, int(D * self.sem_ratio))

        # Combine and project
        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        return self.proj_drop(x)

class Mlp(nn.Module):
    """Multi-Layer Perceptron with GELU activation"""
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
        return self.drop(x)

class STEncoderBlock(nn.Module):
    """Spatio-Temporal Encoder Block"""
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, device=device, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, x_patterns, pattern_keys, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x
        

class STDecoderBlock(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4., 
        qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
        device=torch.device('cpu'), type_ln="pre", output_dim=1
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.self_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, 
            t_num_heads=t_num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
            device=device, output_dim=output_dim
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Cross-attention layer - REMOVED batch_first
        self.norm2 = norm_layer(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=geo_num_heads + sem_num_heads + t_num_heads,
            dropout=attn_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, enc_output, x_patterns, pattern_keys, geo_mask=None, sem_mask=None):
        # Self-attention
        if self.type_ln == 'pre':
            # Self-attention with residual connection
            attn_out = self.self_attn(self.norm1(x), x_patterns, pattern_keys, geo_mask, sem_mask)
            x = x + self.drop_path1(attn_out)
            
            # Cross-attention
            x_normalized = self.norm2(x)
            B, T_dec, F, D = x_normalized.shape
            
            # Flatten spatial and temporal dimensions
            x_flat = x_normalized.reshape(B, T_dec * F, D).permute(1, 0, 2)  # [seq_len, batch, features]
            enc_flat = enc_output.reshape(B, -1, D).permute(1, 0, 2)  # [seq_len, batch, features]
            
            # Cross-attention - no batch_first
            cross_attn_out, _ = self.cross_attn(
                query=x_flat, 
                key=enc_flat, 
                value=enc_flat
            )
            # Convert back to [batch, seq_len, features]
            cross_attn_out = cross_attn_out.permute(1, 0, 2).reshape(B, T_dec, F, D)
            x = x + self.drop_path2(cross_attn_out)
            
            # MLP
            x = x + self.drop_path3(self.mlp(self.norm3(x)))
            
        elif self.type_ln == 'post':
            # Self-attention
            attn_out = self.self_attn(x, x_patterns, pattern_keys, geo_mask, sem_mask)
            x = self.norm1(x + self.drop_path1(attn_out))
            
            # Cross-attention
            B, T_dec, F, D = x.shape
            x_flat = x.reshape(B, T_dec * F, D).permute(1, 0, 2)  # [seq_len, batch, features]
            enc_flat = enc_output.reshape(B, -1, D).permute(1, 0, 2)  # [seq_len, batch, features]
            
            cross_attn_out, _ = self.cross_attn(
                query=x_flat,
                key=enc_flat,
                value=enc_flat
            )
            cross_attn_out = cross_attn_out.permute(1, 0, 2).reshape(B, T_dec, F, D)
            x = self.norm2(x + self.drop_path2(cross_attn_out))
            
            # MLP
            x = self.norm3(x + self.drop_path3(self.mlp(x)))
            
        return x
        
class PDFormerFlow(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # Load data features
        self._scaler = self.data_feature.get('scaler')
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self._logger = getLogger()
        self.dataset = config.get('dataset')
        
        # Flow-specific parameters
        self.num_flows = data_feature['num_flows']
        self.origins = data_feature['origins']  # [num_flows]
        self.destinations = data_feature['destinations']  # [num_flows]
        self.node_features = data_feature['node_features']  # [num_nodes, node_feat_dim]
        
        # Model hyperparameters
        self.embed_dim = config.get('embed_dim', 64)
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 3)
        enc_depth = config.get("enc_depth", 6)
        self.dec_depth = config.get("dec_depth", 3)  # Decoder depth
        type_ln = config.get("type_ln", "pre")
        self.type_short_path = config.get("type_short_path", "dist")
        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5)
        self.dtw_delta = config.get('dtw_delta', 5)
        self.use_curriculum_learning = config.get('use_curriculum_learning', True)
        self.step_size = config.get('step_size', 2500)
        self.max_epoch = config.get('max_epoch', 200)
        self.task_level = config.get('task_level', 0)
        
        # Curriculum learning validation
        if self.max_epoch * self.num_batches * self.world_size < self.step_size * self.output_window:
            self._logger.warning('Parameter `step_size` is too big with {} epochs'.format(self.max_epoch))
        if self.use_curriculum_learning:
            self._logger.info('Using curriculum learning')
        
        # Compute flow-based positional encoding
        self.flow_pos_enc = self._compute_flow_positional_encoding()
        
        # Create flow-based masks
        self.geo_mask, self.sem_mask = self._create_flow_masks(data_feature)
        
        # Initialize pattern keys
        self.pattern_keys = torch.from_numpy(data_feature['flow_pattern_keys']).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])
        
        # Create embedding layers
        self.enc_embed_layer = DataEmbedding(
            feature_dim=self.feature_dim - self.ext_dim,
            embed_dim=self.embed_dim,
            flow_pos_enc=self.flow_pos_enc,
            drop=drop,
            add_time_in_day=add_time_in_day,
            add_day_in_week=add_day_in_week,
            device=self.device
        )
        
        self.dec_embed_layer = DataEmbedding(
            feature_dim=self.feature_dim - self.ext_dim,
            embed_dim=self.embed_dim,
            flow_pos_enc=self.flow_pos_enc,
            drop=drop,
            add_time_in_day=add_time_in_day,
            add_day_in_week=add_day_in_week,
            device=self.device
        )
        
        # Create encoder blocks
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, 
                s_attn_size=self.s_attn_size, 
                t_attn_size=self.t_attn_size, 
                geo_num_heads=geo_num_heads, 
                sem_num_heads=sem_num_heads, 
                t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop, 
                attn_drop=attn_drop, 
                drop_path=enc_dpr[i], 
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                device=self.device, 
                type_ln=type_ln, 
                output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])
        
        # Create decoder blocks
        dec_dpr = [x.item() for x in torch.linspace(0, drop_path, self.dec_depth)]
        self.decoder_blocks = nn.ModuleList([
            STDecoderBlock(
                dim=self.embed_dim, 
                s_attn_size=self.s_attn_size, 
                t_attn_size=self.t_attn_size, 
                geo_num_heads=geo_num_heads, 
                sem_num_heads=sem_num_heads, 
                t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop, 
                attn_drop=attn_drop, 
                drop_path=dec_dpr[i], 
                act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                device=self.device, 
                type_ln=type_ln, 
                output_dim=self.output_dim,
            ) for i in range(self.dec_depth)
        ])
        
        # Prediction head
        self.output_projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.output_dim)
        )

    def _compute_flow_positional_encoding(self):
        """Compute positional encoding for flows by combining origin and destination features"""
        # Convert node features to tensor
        node_feats = torch.from_numpy(self.node_features).float().to(self.device)
        
        # Get features for origins and destinations
        origin_feats = node_feats[self.origins]  # [num_flows, node_feat_dim]
        dest_feats = node_feats[self.destinations]  # [num_flows, node_feat_dim]
        
        # Combine features
        return torch.cat([origin_feats, dest_feats], dim=-1)  # [num_flows, 2*node_feat_dim]
    
    def _create_flow_masks(self, data_feature):
        """Create flow-based geometric and semantic masks"""
        # Load flow distance matrix
        flow_dists = torch.from_numpy(data_feature['flow_dists']).float().to(self.device)
        flow_dtw_matrix = torch.from_numpy(data_feature['flow_dtw_matrix']).float().to(self.device)
        
        # Geometric mask (distance-based)
        if self.type_short_path == "dist":
            # Normalize distances
            valid_dists = flow_dists[~torch.isinf(flow_dists)]
            std = valid_dists.std()
            normalized_dists = flow_dists / std
            
            # Create mask for distant flows
            geo_mask = (normalized_dists > self.far_mask_delta)
        else:
            # Create mask based on hop distance
            geo_mask = (flow_dists > self.far_mask_delta)
        
        # Semantic mask (pattern similarity)
        sem_mask = torch.ones(self.num_flows, self.num_flows, dtype=torch.bool, device=self.device)
        # Find top-k similar flows for each flow
        for i in range(self.num_flows):
            # Get top-k similar flows (excluding self)
            similarities = flow_dtw_matrix[i]
            similarities[i] = float('inf')  # Exclude self
            topk = torch.topk(similarities, self.dtw_delta, largest=False).indices
            sem_mask[i, topk] = False
        
        return geo_mask, sem_mask

    def forward(self, batch):
        x = batch['X']  # Encoder input: [batch, input_window, num_flows, feature_dim]
        if 'future_time' in batch:
            future_time = batch['future_time']
        else:
            # Create dummy future_time tensor
            future_time = torch.zeros(
                x.size(0), 
                self.output_window, 
                self.num_flows, 
                0,  # no temporal features
                device=x.device
            )
        
        # --- Encoder ---
        # Create pattern tensors
        T_enc = x.shape[1]
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T_enc + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # [batch, T_enc, num_flows, s_attn_size, output_dim]
        
        # Embed patterns
        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns_enc = torch.cat(x_pattern_list, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)
        
        # Process through encoder embedding and blocks
        enc = self.enc_embed_layer(x)  # [batch, T_enc, num_flows, embed_dim]
        for encoder_block in self.encoder_blocks:
            enc = encoder_block(enc, x_patterns_enc, pattern_keys, self.geo_mask, self.sem_mask)
        
        # --- Decoder ---
        # Prepare decoder input (future temporal features + zero-valued flow features)
        future_value = torch.zeros(
            future_time.size(0), 
            future_time.size(1), 
            future_time.size(2), 
            self.feature_dim - self.ext_dim,  # Flow feature dimension
            device=self.device
        )
        dec_input = torch.cat([
            future_value,
            future_time[..., :self.ext_dim]  # Temporal features
        ], dim=-1)
        
        # Embed decoder input
        dec = self.dec_embed_layer(dec_input)  # [batch, T_dec, num_flows, embed_dim]
        
        # Create decoder patterns (using zeros since future flows are unknown)
        T_dec = dec_input.shape[1]
        dec_patterns = torch.zeros(
            dec_input.size(0), T_dec, dec_input.size(2), 
            self.s_attn_size, self.output_dim, device=self.device
        )
        dec_pattern_list = []
        for i in range(self.output_dim):
            dec_pattern_list.append(self.pattern_embeddings[i](dec_patterns[..., i]).unsqueeze(-1))
        dec_patterns = torch.cat(dec_pattern_list, dim=-1)
        
        # Process through decoder blocks
        for decoder_block in self.decoder_blocks:
            dec = decoder_block(
                dec, enc, 
                x_patterns=dec_patterns, 
                pattern_keys=pattern_keys,
                geo_mask=self.geo_mask, 
                sem_mask=self.sem_mask
            )
        
        # Output projection
        output = self.output_projection(dec)  # [batch, output_window, num_flows, output_dim]
        return output.permute(0, 1, 3, 2)  # [batch, output_window, output_dim, num_flows]

    def get_loss_func(self, set_loss):
        """Select appropriate loss function"""
        loss_mapping = {
            'mae': loss.masked_mae_torch,
            'mse': loss.masked_mse_torch,
            'rmse': loss.masked_rmse_torch,
            'mape': loss.masked_mape_torch,
            'logcosh': loss.log_cosh_loss,
            'huber': partial(loss.huber_loss, delta=self.huber_delta),
            'quantile': partial(loss.quantile_loss, delta=self.quan_delta),
            'masked_mae': partial(loss.masked_mae_torch, null_val=0),
            'masked_mse': partial(loss.masked_mse_torch, null_val=0),
            'masked_rmse': partial(loss.masked_rmse_torch, null_val=0),
            'masked_mape': partial(loss.masked_mape_torch, null_val=0),
            'masked_huber': partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0),
            'r2': loss.r2_score_torch,
            'evar': loss.explained_variance_score_torch
        }
        
        loss_func = loss_mapping.get(set_loss.lower(), loss.masked_mae_torch)
        if set_loss.lower() not in loss_mapping:
            self._logger.warning(f'Unrecognized loss function {set_loss}, using default MAE')
        return loss_func

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        """Calculate loss between true and predicted values"""
        lf = self.get_loss_func(set_loss)
        # Inverse transform if scaler is available
        if self._scaler:
            y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
            y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
    
        # Curriculum learning scheduling
        if self.training and self.use_curriculum_learning:
            if batches_seen % self.step_size == 0 and self.task_level < self.output_window:
                self.task_level += 1
                self._logger.info(f'Task level increased to {self.task_level} at batch {batches_seen}')
            return lf(y_predicted[:, :self.task_level, ...], y_true[:, :self.task_level, ...])
        else:
            return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None):
        """Calculate loss for a batch"""
        y_true = batch['y']
        y_predicted = self.predict(batch)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)
    
    def predict(self, batch):
        """Generate predictions for a batch"""
        return self.forward(batch)