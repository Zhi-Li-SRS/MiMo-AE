import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# --- Building Blocks: 1D ResNet Block ---
class ResNetBlock1D(nn.Module):
    """
    A 1D Residual Block with Dropout.
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        # Apply ReLU and Dropout after the addition
        return self.dropout(self.relu(out))

# --- Encoder  ---
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, blocks_per_stage, dropout_rate=0.0):
        """
        The encoder for the representation of the input signal.
        """
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        in_ch = base_channels
        for i, num_blocks in enumerate(blocks_per_stage):
            stride = 2
            out_ch = in_ch * 2
            stage = self._make_stage(in_ch, out_ch, num_blocks, stride=stride, dropout_rate=dropout_rate)
            self.layers.append(stage)
            in_ch = out_ch

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = [ResNetBlock1D(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x

# --- Fusion Modules  ---
class ConcatFusion(nn.Module):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        flattened = [torch.flatten(f, 1) for f in features]
        return torch.cat(flattened, dim=1)

class GatedFusion(nn.Module):
    def __init__(self, input_dim, num_modalities):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_modalities),
            nn.Sigmoid()
        )
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        flattened = [torch.flatten(f, 1) for f in features]
        concatenated = torch.cat(flattened, dim=1)
        gating_weights = self.gate(concatenated)
        reshaped_features = torch.stack(flattened, dim=1)
        weighted_features = reshaped_features * gating_weights.unsqueeze(-1)
        return torch.sum(weighted_features, dim=1)

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads, num_modalities):
        super().__init__()
        self.num_modalities = num_modalities
        self.query_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        reshaped_features = torch.stack([torch.flatten(f, 1) for f in features], dim=1)
        batch_size = reshaped_features.size(0)
        query = self.query_token.expand(batch_size, -1, -1)
        fused_feature, _ = self.attention(query, reshaped_features, reshaped_features)
        return fused_feature.squeeze(1)

# --- Decoder  ---
class Decoder(nn.Module):
    def __init__(self, out_channels, base_channels, blocks_per_stage, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        
        in_ch = base_channels * (2 ** len(blocks_per_stage))
        
        for i in reversed(range(len(blocks_per_stage))):
            out_ch = in_ch // 2
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            )
            conv_block = ResNetBlock1D(out_ch, out_ch, dropout_rate=dropout_rate)
            self.layers.append(nn.Sequential(upsample_layer, conv_block))
            in_ch = out_ch
            
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1)
        )
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return self.tanh(x)

# --- Main Multimodal Autoencoder Model ---
class MiMoAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_modalities = config['num_modalities']
        dropout_rate = config.get('dropout_rate', 0.0)
        
        self.encoders = nn.ModuleList([
            Encoder(
                in_channels=config['in_channels'],
                base_channels=config['base_channels'],
                blocks_per_stage=config['blocks_per_stage'],
                dropout_rate=dropout_rate
            ) for _ in range(self.num_modalities)
        ])
        
        encoder_output_len = config['input_len'] // (2 ** (len(config['blocks_per_stage']) + 1))
        encoder_output_channels = config['base_channels'] * (2 ** len(config['blocks_per_stage']))
        compressed_channels = config.get('compressed_channels', 32)

        self.channel_compressor = nn.Conv1d(encoder_output_channels, compressed_channels, kernel_size=1)
        
        feature_dim_per_modality = compressed_channels * encoder_output_len

        fusion_type = config.get('fusion_type', 'concat')
        if fusion_type == 'concat':
            total_feature_dim = feature_dim_per_modality * self.num_modalities
            self.fusion_module = ConcatFusion()
        elif fusion_type == 'gated':
            total_feature_dim = feature_dim_per_modality * self.num_modalities
            self.fusion_module = GatedFusion(total_feature_dim, self.num_modalities)
        elif fusion_type == 'attention':
            total_feature_dim = feature_dim_per_modality
            self.fusion_module = AttentionFusion(feature_dim_per_modality, config['attention_heads'], self.num_modalities)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Adjust GatedFusion output dimension if needed
        if fusion_type == 'gated':
             # GatedFusion with sum aggregation outputs a single modality equivalent dimension
            fused_output_dim = feature_dim_per_modality
        else:
            fused_output_dim = total_feature_dim

        self.to_latent = nn.Linear(fused_output_dim, config['latent_dim'])
        self.fusion_dropout = nn.Dropout(dropout_rate)
        
        self.from_latent = nn.Linear(config['latent_dim'], compressed_channels * encoder_output_len)
        self.latent_dropout = nn.Dropout(dropout_rate)
        self.decoder_reshape_dims = (compressed_channels, encoder_output_len)

        self.channel_expander = nn.Conv1d(compressed_channels, encoder_output_channels, kernel_size=1)

        self.decoders = nn.ModuleList([
            Decoder(
                out_channels=config['in_channels'],
                base_channels=config['base_channels'],
                blocks_per_stage=config['blocks_per_stage'],
                dropout_rate=dropout_rate
            ) for _ in range(self.num_modalities)
        ])

    def forward(self, x):
        modalities = torch.split(x, 1, dim=1)
        
        features = [self.encoders[i](modalities[i]) for i in range(self.num_modalities)]
        
        compressed_features = [self.channel_compressor(f) for f in features]
        
        fused = self.fusion_module(compressed_features)
        fused_dropped = self.fusion_dropout(fused)
        
        latent_vec = self.to_latent(fused_dropped)
        
        x_re = self.from_latent(latent_vec)
        x_re_dropped = self.latent_dropout(x_re)
        
        x_re_reshaped = x_re_dropped.view(-1, *self.decoder_reshape_dims)
        
        x_re_expanded = self.channel_expander(x_re_reshaped)
        
        reconstructions = [self.decoders[i](x_re_expanded) for i in range(self.num_modalities)]
        
        output = torch.cat(reconstructions, dim=1)
        
        if output.shape[-1] != x.shape[-1]:
            output = F.pad(output, (0, x.shape[-1] - output.shape[-1]))

        return output, latent_vec