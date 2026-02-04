from copy import deepcopy
from typing import Optional

from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    GemmaMLP,
    repeat_kv,
)

from .configs import DiTConfig

from torch.nn import RMSNorm
class AdaLayerNormZero(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.silu = nn.SiLU()
        self.linear = nn.Linear(config.dit_hidden_size, 6 * config.dit_hidden_size, bias=True)
        self.norm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        emb = self.linear(self.silu(emb))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        
        # Force float32 for norm
        x_dtype = x.dtype
        x = self.norm(x.to(dtype=torch.float32))
        x = x.to(dtype=x_dtype)

        x = x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    
class AdaLayerNormOut(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        self.silu = nn.SiLU()
        self.linear = nn.Linear(config.dit_hidden_size, config.dit_hidden_size * 2, bias=True)
        self.norm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        
        # Force float32 for norm
        x_dtype = x.dtype
        x = self.norm(x.to(dtype=torch.float32))
        x = x.to(dtype=x_dtype)

        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class DiTSelfAttention(nn.Module):
    """
    Diffusion Transformer Self-Attention.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.num_key_value_groups = config.base_config.num_attention_heads // config.base_config.num_key_value_heads

        if config.qk_norm:
            self.q_norm = RMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)
            self.k_norm = RMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)

        self.q_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_attention_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_key_value_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_key_value_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )

        if config.model_type == "DiT" and config.attention == "self":
            self.text_k_proj = nn.Linear(
                config.dit_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )
            self.text_v_proj = nn.Linear(
                config.dit_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )

        self.o_proj = nn.Linear(config.base_config.num_attention_heads * config.base_config.head_dim, config.dit_hidden_size, bias=config.base_config.attention_bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        pos_embed: Optional[torch.FloatTensor] = None,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None
    ):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.config.attention == "self":
            text_key_states = self.text_k_proj(text_hidden_states)
            text_value_states = self.text_v_proj(text_hidden_states)

        query_states = rearrange(query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
        key_states = rearrange(key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
        value_states = rearrange(value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if self.config.qk_norm:
            dtype = query_states.dtype
            query_states = self.q_norm(query_states.to(dtype=torch.float32)).to(dtype=dtype)
            key_states = self.k_norm(key_states.to(dtype=torch.float32)).to(dtype=dtype)

        if not self.config.pos_embed == "ape":
            cos, sin = pos_embed
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.attention == "self":
            text_key_states = rearrange(text_key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
            text_value_states = rearrange(text_value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

            if self.config.qk_norm:
                dtype = text_key_states.dtype
                text_key_states = self.k_norm(text_key_states.to(dtype=torch.float32)).to(dtype=dtype)

            key_states = torch.cat([text_key_states, key_states], dim=2)
            value_states = torch.cat([text_value_states, value_states], dim=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        attn_output = self.o_proj(attn_output)
        return attn_output


class DiTCrossAttention(nn.Module):
    """
    Diffusion Transformer Cross-Attention.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        self.num_key_value_groups = config.base_config.num_attention_heads // config.base_config.num_key_value_heads

        if config.qk_norm:
            self.q_norm = RMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)
            if config.model_type == "DiT":
                self.k_norm = RMSNorm(config.base_config.head_dim, eps=config.base_config.rms_norm_eps)

        self.q_proj = nn.Linear(
            config.dit_hidden_size,
            config.base_config.num_attention_heads * config.base_config.head_dim,
            bias=config.base_config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.base_config.num_attention_heads * config.base_config.head_dim,
            config.dit_hidden_size,
            bias=config.base_config.attention_bias,
        )

        if config.model_type == "DiT":
            self.text_k_proj = nn.Linear(
                config.text_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )
            self.text_v_proj = nn.Linear(
                config.text_hidden_size,
                config.base_config.num_key_value_heads * config.base_config.head_dim,
                bias=config.base_config.attention_bias,
            )

    def forward(
        self,
        dit_hidden_states: torch.FloatTensor,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        key_states: Optional[torch.FloatTensor] = None,
        value_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        # Step 1: validate inputs
        #if torch.isnan(dit_hidden_states).any():
        #    raise ValueError(f"[CrossAttn] NaN in dit_hidden_states input")
        #if text_hidden_states is not None and torch.isnan(text_hidden_states).any():
        #    raise ValueError(f"[CrossAttn] NaN in text_hidden_states input")
        
        # Step 2: Q projection
        query_states = self.q_proj(dit_hidden_states)
        #if torch.isnan(query_states).any():
        #    raise ValueError(f"[CrossAttn] NaN after q_proj")
        
        # Step 3: K/V projection
        if self.config.model_type == "DiT":
            key_states = self.text_k_proj(text_hidden_states)
            value_states = self.text_v_proj(text_hidden_states)
            #if torch.isnan(key_states).any():
            #    raise ValueError(f"[CrossAttn] NaN after text_k_proj")
            #if torch.isnan(value_states).any():
            #    raise ValueError(f"[CrossAttn] NaN after text_v_proj")

        # Step 4: Reshape
        query_states = rearrange(query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
        if self.config.model_type == "DiT":
            key_states = rearrange(key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
            value_states = rearrange(value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        # Step 5: QK Norm
        if self.config.qk_norm:
            dtype = query_states.dtype
            query_states = self.q_norm(query_states.to(dtype=torch.float32)).to(dtype=dtype)
            #if torch.isnan(query_states).any():
            #    raise ValueError(f"[CrossAttn] NaN after q_norm")
            if self.config.model_type == "DiT":
                key_states = self.k_norm(key_states.to(dtype=torch.float32)).to(dtype=dtype)
            #    if torch.isnan(key_states).any():
            #        raise ValueError(f"[CrossAttn] NaN after k_norm")
        
        # Step 6: Repeat KV
        if self.config.model_type == "DiT":
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Step 7: sanity-check attention_mask
        #if attention_mask is not None:
        #    if torch.isnan(attention_mask).any():
        #        raise ValueError(f"[CrossAttn] NaN in attention_mask")
            # Detect rows that are fully masked (-inf) which would create NaNs
        #    if attention_mask.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        #        all_masked = (attention_mask < -1e9).all(dim=-1)
        #        if all_masked.any():
        #            raise ValueError(f"[CrossAttn] attention_mask has rows that are all masked (-inf), will cause NaN in softmax")

        # Step 8: inspect QK dot-product range (debug)
        #with torch.no_grad():
        #    qk_scale = query_states.shape[-1] ** -0.5
        #    qk = torch.matmul(query_states, key_states.transpose(-2, -1)) * qk_scale
        #    if torch.isnan(qk).any():
        #        raise ValueError(f"[CrossAttn] NaN in QK dot product")
        #    qk_max = qk.abs().max().item()
        #    if qk_max > 100:
        #        print(f"⚠️ [CrossAttn] Warning: QK values too large: max={qk_max:.2f}")

        # Step 9: Scaled Dot Product Attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, 
            attn_mask=attention_mask, is_causal=False
        )
        #if torch.isnan(attn_output).any():
        #    raise ValueError(f"[CrossAttn] NaN after scaled_dot_product_attention")

        # Step 10: reshape and O projection
        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        attn_output = self.o_proj(attn_output)
        #if torch.isnan(attn_output).any():
        #    raise ValueError(f"[CrossAttn] NaN after o_proj")

        return attn_output


class DiTLayer(nn.Module):
    """
    Diffusion Transformer Block.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        if config.timestep_conditioning == "adaln-zero":
            self.input_layernorm = AdaLayerNormZero(config)
        else:
            if config.timestep_conditioning == "adaln-single":
                self.scale_shift_table = nn.Parameter(torch.randn(6 * config.dit_hidden_size) / config.dit_hidden_size ** 0.5)
            self.input_layernorm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)

        self.self_attn = DiTSelfAttention(config)
        if config.attention == "cross":
            self.cross_attn = DiTCrossAttention(config)

        self.post_attention_layernorm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        if config.sandwich_norm:
            self.pre_feedforward_layernorm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
            self.post_feedforward_layernorm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
            if config.attention == "cross":
                self.post_cross_attention_layernorm = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        mlp_config = deepcopy(config.base_config)
        mlp_config.hidden_size = config.dit_hidden_size
        mlp_config.intermediate_size = config.dit_hidden_size * 4
        self.mlp = GemmaMLP(mlp_config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        pos_embed: Optional[torch.FloatTensor] = None,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        if self.config.timestep_conditioning == "adaln-zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.input_layernorm(hidden_states, emb=temb)
            shift_msa = scale_msa = torch.zeros_like(temb)
        else:
            dtype = hidden_states.dtype
            norm_hidden_states = self.input_layernorm(hidden_states.to(dtype=torch.float32)).to(dtype=dtype)
            if self.config.timestep_conditioning == "adaln-single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table + temb).chunk(6, dim=1)
            else:
                shift_msa = scale_msa = shift_mlp = scale_mlp = torch.zeros_like(temb)
                gate_msa = gate_mlp = torch.ones_like(temb)

        # Attention.
        #if torch.isnan(norm_hidden_states).any():
        #    raise ValueError("NaN values found in norm_hidden_states before attention")
        norm_hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]

        if self.config.attention == "self":
            attn_output = self.self_attn(norm_hidden_states, pos_embed, text_hidden_states, attention_mask)

            if self.config.sandwich_norm:
                dtype = attn_output.dtype
                attn_output = self.post_attention_layernorm(attn_output.to(dtype=torch.float32)).to(dtype=dtype)

            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        elif self.config.attention == "cross":
            attn_output = self.self_attn(norm_hidden_states, pos_embed)
            #if torch.isnan(attn_output).any():
            #    raise ValueError("NaN values found in attn_output after self attention")
            if self.config.sandwich_norm:
                dtype = attn_output.dtype
                attn_output = self.post_attention_layernorm(attn_output.to(dtype=torch.float32)).to(dtype=dtype)

            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
            #if torch.isnan(hidden_states).any():
            #    raise ValueError("NaN values found in hidden_states before cross attention")
            cross_attn_output = self.cross_attn(hidden_states, text_hidden_states=text_hidden_states, attention_mask=attention_mask)
            #if torch.isnan(cross_attn_output).any():
            #    raise ValueError("NaN values found in cross_attn_output after cross attention")
            if self.config.sandwich_norm:
                dtype = cross_attn_output.dtype
                cross_attn_output = self.post_cross_attention_layernorm(cross_attn_output.to(dtype=torch.float32)).to(dtype=dtype)

            hidden_states = hidden_states + cross_attn_output

        if self.config.sandwich_norm:
            dtype = hidden_states.dtype
            norm_hidden_states = self.pre_feedforward_layernorm(hidden_states.to(dtype=torch.float32)).to(dtype=dtype)
        else:
            dtype = hidden_states.dtype
            norm_hidden_states = self.post_attention_layernorm(hidden_states.to(dtype=torch.float32)).to(dtype=dtype)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        #if torch.isnan(norm_hidden_states).any():
        #    raise ValueError("NaN values found in norm_hidden_states before MLP")
        mlp_output = self.mlp(norm_hidden_states)
        #if torch.isnan(mlp_output).any():
        #    raise ValueError("NaN values found in mlp_output after MLP")
        if self.config.sandwich_norm:
            dtype = mlp_output.dtype
            mlp_output = self.post_feedforward_layernorm(mlp_output.to(dtype=torch.float32)).to(dtype=dtype)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * mlp_output

        return hidden_states
