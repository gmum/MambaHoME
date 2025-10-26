from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from src.models.soft_moe_2d_3d import HierarchicalSoftMoE2DBlock


class DynamicTanh(nn.Module):
    """
    A LayerNorm-like module implemented as a learnable tanh-based transform.

    Args:
        normalized_shape: int or tuple - number of features to normalize
        alpha_init_value: float - initial value for alpha scaling the tanh
    """
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        # Per-channel learnable scale and bias
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        alpha_clamped = self.alpha.clamp(0.1, 2.0)
        return self.weight * torch.tanh(alpha_clamped * x) + self.bias

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}"


class MambaBlock(nn.Module):
    """
    Wrapper around Mamba SSM block with DynamicTanh normalization.

    Args:
        dim: int - token feature dimension (d_model)
        d_state: int - internal Mamba state dimension (controls memory)
        d_conv: int - Mamba local convolution kernel / conv-width
        expand: int - expansion factor in Mamba feedforward-like path
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = DynamicTanh(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x_flat):
        # x_flat: [B, N, C]
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        return x_mamba


class GSC(nn.Module):
    """
    Global Spatial Convolutional module.
    Combines parallel convolutions (3x3 branch and 1x1 branch) with instance norm and ReLU,
    followed by fusion and residual connection.

    - proj: Conv3d(in_channels -> in_channels, kernel_size=3, stride=1, padding=1)
    - proj2: Conv3d(in_channels -> in_channels, kernel_size=3, stride=1, padding=1)
    - proj3: Conv3d(in_channels -> in_channels, kernel_size=1, stride=1, padding=0)
    - proj4: Conv3d(in_channels -> in_channels, kernel_size=1, stride=1, padding=0)
    """
    def __init__(self, in_channels):
        super().__init__()
        # 3x3 branch
        self.proj = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.nonliner = nn.ReLU()
        self.proj2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.nonliner2 = nn.ReLU()
        # 1x1 branch (preserve low-level details)
        self.proj3 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.nonliner3 = nn.ReLU()
        # fusion 1x1
        self.proj4 = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)
        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        return x + x_residual


class MambaEncoder(nn.Module):
    """
    Mamba-based hierarchical encoder with Hierarchical Soft MoE layers.

    Args:
        in_chans: int = 1
            Number of input channels (e.g., 1 for grayscale CT).
        depths: list[int] = [2,2,2,2]
            Number of Mamba+MoE blocks per stage (4 stages total).
        dims: list[int] = [48,96,192,384]
            Channel dimension at each stage.
        drop_path_rate: float = 0.0
            (Optional) drop path regularization rate.
        layer_scale_init_value: float = 1e-6
            (Optional) residual layer scale initialization (not used explicitly here).
        out_indices: list[int] = [0,1,2,3]
            Indices of stages to return as outputs for skip connections.
        num_experts: int = 16
            Base number of experts (not used directly; stage-specific lists are used).
        expert_mult: int = 4
            Multiplicative factor to determine expert hidden dims.
        moe_dropout: float = 0.0
            Dropout rate inside the MoE blocks.
        use_geglu: bool = True
            Use GEGLU gating inside expert MLPs if True.
        group_size: int = 16
            Default tokens-per-group (can be overridden with group_list).
        num_slots_per_expert_first: int = 4
            Number of slots assigned to each first-level expert.
        experts_list: list[int] = [4,8,12,16]
            Number of first-level experts per stage (length must be 4).
        experts_list_second: list[int] = [8,16,24,32]
            Number of second-level experts per stage (length must be 4).
        group_list: list[int] = [2048,1024,512,256]
            Token grouping sizes per stage (length must be 4).
    """
    def __init__(
        self,
        in_chans=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        out_indices=[0, 1, 2, 3],
        expert_mult=4,
        moe_dropout=0.0,
        use_geglu=True,
        num_slots_per_expert_first=4,
        experts_list=[4, 8, 12, 16],
        experts_list_second=[8, 16, 24, 32],
        group_list=[2048, 1024, 512, 256],
    ):
        super().__init__()
        assert len(depths) == 4 and len(dims) == 4, "depths and dims must contain 4 elements (4 stages)"
        assert len(experts_list) == 4 and len(experts_list_second) == 4 and len(group_list) == 4, \
            "experts_list, experts_list_second and group_list must be length 4 (per-stage values)"

        self.depths = depths
        self.experts_list = experts_list
        self.experts_list_second = experts_list_second
        self.group_list = group_list
        self.dims = dims
        self.out_indices = out_indices

        # -----------------------
        # Downsampling / stem
        # -----------------------
        # stem conv3d: kernel_size=7, stride=2, padding=3 (reduces spatial resolution by 2)
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers = nn.ModuleList([stem])

        # subsequent downsampling layers per stage: Conv3d with kernel_size=2, stride=2
        # (each halves spatial resolution and projects channels dims[i] -> dims[i+1])
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(dims[i]),
                    nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),  # kernel=2, stride=2 -> halve
                )
            )

        # -----------------------
        # Stage-wise modules
        # -----------------------
        self.gscs = nn.ModuleList()
        self.mamba_blocks = nn.ModuleList()
        self.moe_layers = nn.ModuleList()

        # Norms applied before SMoE (DynamicTanh)
        self.norm_moe = nn.ModuleList([
            nn.ModuleList([DynamicTanh(dims[i]) for _ in range(depths[i])])
            for i in range(4)
        ])

        for i in range(4):
            # GSC block for local-global context:
            gsc = GSC(dims[i])

            # MambaBlocks repeated 'depths[i]' times.
            mamba_blocks = nn.ModuleList([
                MambaBlock(dim=dims[i]) for _ in range(depths[i])
            ])

            # Hierarchical SMoE blocks repeated per depth:
            moe_layers = nn.ModuleList([
                HierarchicalSoftMoE2DBlock(
                    dim=dims[i],
                    group_size=self.group_list[i],  # token group size for this stage
                    num_experts_first=self.experts_list[i],  # first-level experts count
                    num_slots_per_expert_first=num_slots_per_expert_first,
                    num_experts_second=self.experts_list_second[i],  # second-level experts count
                    expert_mult=expert_mult,  # expert internal expansion multiplier
                    dropout=moe_dropout,
                    use_geglu=use_geglu,
                ) for _ in range(depths[i])
            ])

            self.gscs.append(gsc)
            self.mamba_blocks.append(mamba_blocks)
            self.moe_layers.append(moe_layers)

        # final instance norms for each output stage
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            setattr(self, f"norm{i_layer}", layer)

    def forward_features(self, x):
        """
        Compute multi-scale features from input volume x.

        Args:
            x: Tensor [B, C_in, D, H, W]
        Returns:
            tuple of feature maps (B, C_stage, D_stage, H_stage, W_stage) for stages in out_indices
        """
        outs = []
        for i in range(4):
            # Downsample / stem for stage i
            x = self.downsample_layers[i](x)  # conv3d downsample, preserves channel dims[i]

            # GSC: local-global context
            x = self.gscs[i](x)

            B, C, D, H, W = x.shape
            # Flatten spatial tokens into sequence for SSM / MoE: [B, N, C] where N = D*H*W
            x_flat = x.reshape(B, C, -1).transpose(-1, -2)  # [B, N, C]

            # Per-depth blocks: Mamba (SSM) followed by Hierarchical MoE
            for j in range(self.depths[i]):
                # MambaBlock expects token sequence [B,N,C] and returns [B,N,C]
                x_flat = self.mamba_blocks[i][j](x_flat) + x_flat

                # Normalize with DynamicTanh then pass to Hierarchical SMoE block
                x_flat = self.moe_layers[i][j](self.norm_moe[i][j](x_flat)) + x_flat

            # Reshape back to 3D volume
            x = x_flat.transpose(-1, -2).reshape(B, C, D, H, W)

            # Optionally return stage outputs for skip connections
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                outs.append(norm_layer(x))

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


class MambaHoME(nn.Module):
    """
    Full encoder-decoder model combining MambaEncoder backbone with UNETR encoder/decoder
    and final UnetOutBlock projection.

    Args:
        in_chans: int = 1
            input channels
        out_chans: int = 13
            segmentation output channels/classes
        depths, feat_size: lists
            forwarded to MambaEncoder (see MambaEncoder docs)
        drop_path_rate, layer_scale_init_value: forwarded
        hidden_size: int = 768
            bottleneck channel dimension used by encoder5 / decoder5
        norm_name: str = "instance"
            normalization name passed to UnetrBasicBlock / UnetrUpBlock (e.g. "instance", "batch")
        conv_block: bool = True
            whether to use conv blocks in UNETR blocks (parameter forwarded to UnetrBasicBlock)
        res_block: bool = True
            whether to use residual blocks inside UNETR blocks
        spatial_dims: int = 3
            spatial dimensionality (3 for volumetric)
        num_experts, expert_mult, moe_dropout, use_geglu, group_size, num_slots_per_expert_first,
        experts_list, experts_list_second, group_list:
            forwarded to MambaEncoder (see MambaEncoder docs)
    """
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        norm_name="instance",
        conv_block=True,
        res_block=True,
        spatial_dims=3,
        expert_mult=2,
        moe_dropout=0.0,
        use_geglu=True,
        num_slots_per_expert_first=4,
        experts_list=[4, 8, 12, 16],
        experts_list_second=[8, 16, 24, 32],
        group_list=[2048, 1024, 512, 256],
    ):
        super().__init__()
        assert len(depths) == 4 and len(feat_size) == 4, "depths and feat_size must be length 4"

        self.hidden_size = hidden_size
        self.feat_size = feat_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.spatial_dims = spatial_dims

        # -----------------------
        # Encoder backbone (Mamba + Hierarchical SMoE)
        # -----------------------
        self.mamba_encoder = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            expert_mult=expert_mult,
            moe_dropout=moe_dropout,
            use_geglu=use_geglu,
            num_slots_per_expert_first=num_slots_per_expert_first,
            experts_list=experts_list,
            experts_list_second=experts_list_second,
            group_list=group_list,
            out_indices=[0, 1, 2, 3],
        )

        # -----------------------
        # UNETR-style encoder blocks (project features for skip connections)
        # Each UnetrBasicBlock uses:
        #   kernel_size=3, stride=1 -> preserves resolution
        #   norm_name: e.g. "instance"
        #   res_block: use residual inside block
        # -----------------------
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        # bottleneck projection: from feat_size[3] -> hidden_size
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # -----------------------
        # Decoder / upsampling blocks
        # Each UnetrUpBlock:
        #   kernel_size=3 for local refinement after upsample
        #   upsample_kernel_size=2 doubles spatial resolution
        # -----------------------
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        # final local refinement (no upsampling)
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # -----------------------
        # Output projection
        # UnetOutBlock typically: Conv3d(in_channels=feat_size[0], out_channels=out_chans, kernel_size=1)
        # -----------------------
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.out_chans,
        )

    def forward(self, x_in):
        """
        Forward pass.

        Args:
            x_in: Tensor [B, in_chans, D, H, W]

        Returns:
            Tensor [B, out_chans, D, H, W] segmentation logits (same spatial resolution as input if sizes align)
        """
        # MambaEncoder returns tuple of stage outputs (x2, x3, x4, x5)
        outs = self.mamba_encoder(x_in)
        # shallow convolutional encoder applied directly to input
        enc1 = self.encoder1(x_in)

        # Unpack encoder outputs from the Mamba encoder:
        # outs correspond to stage outputs at progressively lower resolutions:
        # x2 (stage0 output), x3 (stage1 output), x4 (stage2 output), x5 (stage3 output)
        x2, x3, x4, x5 = outs

        enc2 = self.encoder2(x2)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        # enc_hidden is bottleneck representation used by decoder5
        enc_hidden = self.encoder5(x5)

        # Decoder: each UnetrUpBlock upsamples its input, merges with encoder skip and refines
        dec3 = self.decoder5(enc_hidden, enc4)  # upsamples hidden -> feat_size[3], fuse with enc4
        dec2 = self.decoder4(dec3, enc3)        # upsamples -> feat_size[2], fuse with enc3
        dec1 = self.decoder3(dec2, enc2)        # upsamples -> feat_size[1], fuse with enc2
        dec0 = self.decoder2(dec1, enc1)        # upsamples -> feat_size[0], fuse with enc1
        out = self.decoder1(dec0)               # local refine

        return self.out(out)  # project to out_chans