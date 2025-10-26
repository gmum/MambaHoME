from torch import nn
import einops
from src.models.soft_moe.soft_moe_with_dynamic_slots import HierarchicalDynamicSlotsSoftMoE


class HierarchicalSoftMoE2DBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            group_size: int,
            num_experts_first: int,
            num_slots_per_expert_first: int,
            num_experts_second: int,
            expert_mult: int = 4,
            dropout: float = 0.0,
            use_geglu: bool = True,
    ):
        super().__init__()
        self.mlp = HierarchicalDynamicSlotsSoftMoE(
            dim=dim,
            group_size=group_size,
            num_experts_first=num_experts_first,
            num_slots_per_expert_first=num_slots_per_expert_first,
            num_experts_second=num_experts_second,
            expert_mult=expert_mult,
            dropout=dropout,
            geglu=use_geglu,
        )

    def forward(self, x):

        out = self.mlp(x)

        return out
