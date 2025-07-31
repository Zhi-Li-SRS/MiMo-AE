import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# --- Building Blocks: 1D ResNet Block ---
class ResNetBlock1D(nn.Module):
    """
    A 1D Residual Block.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        # x shape: (bsz, in_channels, seq_len)
        identity = self.shortcut(x) # 
        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        out += identity # (bsz, out_channels, seq_len)
        return self.relu(out)
    
    
# --- Encoder  ---
class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, blocks_per_stage):
        """
        The encoder for the representation of the input signal.

        Args:
            in_channels (int): The number of channels in the input signal.
            base_channels (int): The number of channels in the base layer.
            blocks_per_stage (list): The number of blocks in each stage.
        """
        super().__init__()
        # Downsampling shape is (bsz, in_channels, seq_len) -> (bsz, base_channels, seq_len/2)
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
            stage = self._make_stage(in_ch, out_ch, num_blocks, stride=stride)
            self.layers.append(stage)
            in_ch = out_ch

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = [ResNetBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        return x

# --- Fusion Modules  ---
class ConcatFusion(nn.Module):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features shape: (bsz, num_modalities (3), feature_dim)
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
    def __init__(self, out_channels, base_channels, blocks_per_stage, initial_len):
        super().__init__()
        self.layers = nn.ModuleList()
        
        in_ch = base_channels * (2 ** len(blocks_per_stage))
        
        for i in reversed(range(len(blocks_per_stage))):
            out_ch = in_ch // 2
            # Upsampling using ConvTranspose1d              
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            )
            conv_block = ResNetBlock1D(out_ch, out_ch)
            self.layers.append(nn.Sequential(upsample_layer, conv_block))
            in_ch = out_ch
            
        # Final upsampling and output convolution
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1)
        )
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=7, padding=3)
        self.tanh = nn.Tanh() # To bound the output, often helps reconstruction

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

        # Create encoders for each modality
        self.encoders = nn.ModuleList([
            Encoder(
                in_channels=config['in_channels'],
                base_channels=config['base_channels'],
                blocks_per_stage=config['blocks_per_stage']
            ) for _ in range(self.num_modalities)
        ])
        
        # Calculate feature dimension before fusion
        encoder_output_len = config['input_len'] // (2 ** (len(config['blocks_per_stage']) + 1))
        encoder_output_channels = config['base_channels'] * (2 ** len(config['blocks_per_stage']))
        self.feature_dim_per_modality = encoder_output_channels * encoder_output_len

        # Instantiate the chosen fusion module
        fusion_type = config.get('fusion_type', 'concat')
        if fusion_type == 'concat':
            total_feature_dim = self.feature_dim_per_modality * self.num_modalities
            self.fusion_module = ConcatFusion()
        elif fusion_type == 'gated':
            total_feature_dim = self.feature_dim_per_modality
            self.fusion_module = GatedFusion(self.feature_dim_per_modality * self.num_modalities, self.num_modalities)
        elif fusion_type == 'attention':
            total_feature_dim = self.feature_dim_per_modality
            self.fusion_module = AttentionFusion(self.feature_dim_per_modality, config['attention_heads'], self.num_modalities)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.to_latent = nn.Linear(total_feature_dim, config['latent_dim'])
        
        self.from_latent = nn.Linear(config['latent_dim'], encoder_output_channels * encoder_output_len)
        self.decoder_reshape_dims = (encoder_output_channels, encoder_output_len)

        # Create simple decoders for each modality (no skip connections)
        self.decoders = nn.ModuleList([
            Decoder(
                out_channels=config['in_channels'],
                base_channels=config['base_channels'],
                blocks_per_stage=config['blocks_per_stage'],
                initial_len=encoder_output_len
            ) for _ in range(self.num_modalities)
        ])

    def forward(self, x):
        # x shape: (bsz, num_modalities (3), seq_len (240))
        modalities = torch.split(x, 1, dim=1) # split into 3 modalities (bsz, 1, 240)
        
        # --- ENCODING ---
        features = [self.encoders[i](modalities[i]) for i in range(self.num_modalities)]

        # --- FUSION & BOTTLENECK ---
        fused = self.fusion_module(features)
        
        latent_vec = self.to_latent(fused)
        
        # --- DECODING ---
        x_re = self.from_latent(latent_vec)
        x_re = x_re.view(-1, *self.decoder_reshape_dims)
        
        reconstructions = [self.decoders[i](x_re) for i in range(self.num_modalities)]
        
        output = torch.cat(reconstructions, dim=1)
        
        # Ensure output length matches input length
        if output.shape[-1] != x.shape[-1]:
            output = F.pad(output, (0, x.shape[-1] - output.shape[-1]))

        return output, latent_vec

