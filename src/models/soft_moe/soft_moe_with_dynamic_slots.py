import math
import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F
from einops import rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange



def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def l2norm(t):
    return F.normalize(t, dim=-1)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return True, F.pad(tensor, (*pad_offset, 0, remainder), value=value)


class RMSNorm(Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1, eps=self.eps) * self.scale * self.gamma


def FeedForward(dim, mult=4, dropout=0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )


class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


def GLUFeedForward(dim, mult=4, dropout=0.):
    dim_hidden = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim)
    )


class DynamicSlotsSoftMoE(Module):
    def __init__(
            self,
            dim,
            *,
            num_experts=4,
            expert_mult=4,
            dropout=0.,
            geglu=False
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.num_experts = num_experts
        self.to_slot_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_experts, bias=False),
            Rearrange('b n (e d) -> b e n d', e=num_experts),
            RMSNorm(dim)
        )
        expert_class = GLUFeedForward if geglu else FeedForward
        self.experts = nn.ModuleList([
            expert_class(dim=dim, mult=expert_mult, dropout=dropout) for _ in range(num_experts)
        ])

    def forward(self, x, mask=None):
        seq_len, is_image, num_experts = x.shape[-2], x.ndim == 4, self.num_experts
        if is_image:
            x = rearrange(x, 'b d h w -> b h w d')
            x, ps = pack([x], 'b * d')
        x = self.norm(x)
        is_padded, x = pad_to_multiple(x, num_experts, dim=-2)
        if is_padded:
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
            _, mask = pad_to_multiple(mask, num_experts, dim=-1, value=False)
        x_segmented = rearrange(x, 'b (n e) d -> b n e d', e=num_experts)
        if exists(mask):
            segmented_mask = rearrange(mask, 'b (n e) -> b n e', e=num_experts)
            x_segmented = x_segmented.masked_fill(~rearrange(segmented_mask, '... -> ... 1'), 0.)
        if exists(mask):
            num = reduce(x_segmented, 'b n e d -> b n d', 'sum')
            den = reduce(segmented_mask.float(), 'b n e -> b n 1', 'sum').clamp(min=1e-5)
            x_consecutive_mean = num / den
            slots_mask = segmented_mask.any(dim=-1)
        else:
            x_consecutive_mean = reduce(x_segmented, 'b n e d -> b n d', 'mean')
        slot_embeds = self.to_slot_embeds(x_consecutive_mean)
        logits = einsum('b n d, b e s d -> b n e s', x, slot_embeds)
        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1 1')
            slots_mask = rearrange(slots_mask, 'b s -> b 1 1 s')
            logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
            logits = logits.masked_fill(~slots_mask, -torch.finfo(logits.dtype).max)
        dispatch_weights = logits.softmax(dim=1)
        combine_weights = rearrange(logits, 'b n e s -> b n (e s)')
        combine_weights = combine_weights.softmax(dim=-1)
        slots = einsum('b n d, b n e s -> e b s d', x, dispatch_weights)
        out = [expert(slots_per_expert) for slots_per_expert, expert in zip(slots, self.experts)]
        out = torch.stack(out)
        out = rearrange(out, 'e b s d -> b (e s) d')
        out = einsum('b s d, b n s -> b n d', out, combine_weights)
        if is_image:
            out, = unpack(out, ps, 'b * d')
            out = rearrange(out, 'b h w d -> b d h w')
        return out[:, :seq_len]


class HierarchicalDynamicSlotsSoftMoE(Module):
    def __init__(
            self,
            dim,
            group_size,  # Number of tokens per group (K)
            num_experts_first,  # Number of experts in the first level (E1)
            num_slots_per_expert_first,  # Number of slots per expert per group (S)
            num_experts_second,  # Number of experts in the second level (E2)
            expert_mult=4,
            dropout=0.,
            geglu=False
    ):
        super().__init__()
        self.dim = dim
        self.group_size = group_size
        self.num_experts_first = num_experts_first
        self.num_slots_per_expert_first = num_slots_per_expert_first

        self.norm = RMSNorm(dim)
        self.to_slot_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_experts_first * num_slots_per_expert_first, bias=False),
            Rearrange('b g (e s d) -> b g e s d', e=num_experts_first, s=num_slots_per_expert_first),
            RMSNorm(dim)
        )

        # Experts for the first level
        expert_class = GLUFeedForward if geglu else FeedForward
        self.experts_first = nn.ModuleList([
            expert_class(dim=dim, mult=expert_mult, dropout=dropout)
            for _ in range(num_experts_first)
        ])

        # Adaptive gating per expert
        self.expert_gates = nn.Parameter(torch.ones(num_experts_first))  # shape: (E,)

        # Group-level gating
        self.group_gate_mlp = nn.Linear(dim, 1)

        # Second level DynamicSlotsSoftMoE
        self.dynamic_slots_soft_moe = DynamicSlotsSoftMoE(
            dim=dim,
            num_experts=num_experts_second,
            expert_mult=expert_mult,
            dropout=dropout,
            geglu=geglu
        )

    def forward(self, x, mask=None):
        b, N, d = x.shape
        K = self.group_size
        E = self.num_experts_first
        S = self.num_slots_per_expert_first
        M = E * S

        # Pad sequence to be divisible by group_size
        is_padded, x_padded = pad_to_multiple(x, K, dim=-2)
        if is_padded:
            pad_length = x_padded.shape[1] - N
            if exists(mask):
                mask = F.pad(mask, (0, pad_length), value=False)
            else:
                mask = torch.ones((b, N), device=x.device, dtype=torch.bool)
                mask = F.pad(mask, (0, pad_length), value=False)
        else:
            mask = torch.ones((b, N), device=x.device, dtype=torch.bool) if not exists(mask) else mask

        G = x_padded.shape[1] // K
        x_groups = rearrange(x_padded, 'b (G K) d -> b G K d', K=K)
        mask_groups = rearrange(mask, 'b (G K) -> b G K', K=K)

        # Normalize input for slot assignment
        x_norm = self.norm(x_groups)  # (b, G, K, d)

        # Compute slot embeddings dynamically
        group_means = x_groups.mean(dim=2)  # (b, G, d)
        slot_embeds = self.to_slot_embeds(group_means)  # (b, G, E, S, d)

        # Slot assignment
        logits = einsum('b G K d, b G E S d -> b G K E S', x_norm, slot_embeds)
        logits = logits.masked_fill(
            ~rearrange(mask_groups, 'b G K -> b G K 1 1'),
            -torch.finfo(logits.dtype).max
        )
        combine_weights = rearrange(logits, 'b G K E S -> b G K (E S)').softmax(dim=-1)
        slots = einsum('b G K d, b G K M -> b G M d', x_groups, combine_weights)

        # Group gating weights
        group_features = x_groups.mean(dim=2)  # (b, G, d)
        group_gates = torch.sigmoid(self.group_gate_mlp(group_features)).clamp(0.01, 1.0)  # (b, G, 1)

        # Process slots with experts + adaptive gating
        out_first = []
        for e in range(E):
            slots_e = slots[:, :, e * S: (e + 1) * S, :]  # (b, G, S, d)
            slots_e = rearrange(slots_e, 'b G S d -> (b G) S d')
            out_e = self.experts_first[e](slots_e) * self.expert_gates[e]
            out_e = rearrange(out_e, '(b G) S d -> b G S d', b=b)
            out_first.append(out_e)

        out_first = torch.cat(out_first, dim=2)  # (b, G, M, d)
        out_first = out_first * group_gates.unsqueeze(2)  # (b, G, M, d)

        # Second-level SMoE
        new_sequence = rearrange(out_first, 'b G M d -> b (G M) d')
        out_second = self.dynamic_slots_soft_moe(new_sequence)
        out_second_groups = rearrange(out_second, 'b (G M) d -> b G M d', G=G)

        # Map back to token space
        final_out_groups = einsum('b G M d, b G K M -> b G K d', out_second_groups, combine_weights)
        final_out = rearrange(final_out_groups, 'b G K d -> b (G K) d')
        final_out = final_out[:, :N, :]  # remove padding if any

        return final_out
