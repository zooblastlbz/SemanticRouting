import functools
import math
from typing import Optional
from copy import deepcopy


from diffusers.models.embeddings import PatchEmbed, Timesteps
from diffusers.utils import is_torch_xla_available
from einops import rearrange, repeat
import torch
from torch import ge, nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaModel,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma2Model,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    AutoModelForImageTextToText
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
    repeat_kv,
)
from torch.nn import RMSNorm


from .configs import DiTConfig, FuseDiTConfig
from .modules import AdaLayerNormOut, DiTLayer

if is_torch_xla_available():
    ACCEL = "xla"
elif torch.cuda.is_available():
    ACCEL = "cuda"
else:
    ACCEL = "cpu"


class TimestepWiseFeatureWeighting(nn.Module):
    """
    MLP-based timestep-aware feature weighting module.

    It receives a timestep t and a set of features from different text encoder layers,
    then learns weights via a small MLP to combine them into a single representation.
    """
    def __init__(self, num_layers: int, time_embed_dim: int, feature_dim: int):
        """
        Initialize the module.

        Args:
            num_layers (int): Number of feature layers to combine.
            time_embed_dim (int): Dimension of the timestep embedding vector.
            feature_dim (int): Feature dimension for layer norm.
        """
        super().__init__()
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.feature_dim = feature_dim

        # MLP to generate weights
        hidden_dim = time_embed_dim * 4
        self.weight_generator = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_layers),
        )

        # Zero-init final layer so softmax starts near uniform
        nn.init.zeros_(self.weight_generator[-1].weight)
        nn.init.zeros_(self.weight_generator[-1].bias)

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Encode timestep t with sinusoidal embedding.
        """
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        )
        
        args = timesteps.unsqueeze(-1).float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.time_embed_dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

    def forward(self, hidden_states: list, timesteps: torch.Tensor):
        """
        Forward pass for weighting.
        """
        if len(hidden_states) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} hidden states, but got {len(hidden_states)}")

        dtype = hidden_states[0].dtype

        # F.layer_norm handles AMP precision automatically
        normalized_hidden_states = [
            F.layer_norm(h, normalized_shape=(self.feature_dim,))
            for h in hidden_states
        ]

        stacked_hidden_states = torch.stack(normalized_hidden_states, dim=0)

        # Timestep embedding
        t_embed = self._time_embedding(timesteps)
        weight_logits = self.weight_generator(t_embed.to(dtype))
        # F.softmax handles AMP precision automatically
        weights = F.softmax(weight_logits, dim=-1)

        # Weighted sum
        broadcastable_weights = weights.T.unsqueeze(-1).unsqueeze(-1)
        weighted_states = stacked_hidden_states * broadcastable_weights
        combined_features = torch.sum(weighted_states, dim=0)

        return combined_features, weights


def get_llm(model: str, base_config: PretrainedConfig):

        # Fallbacks to support more encoder families, e.g., Qwen/Qwen-VL
        # 1) Try pure causal LM first
    try:
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(model)
        return lm.model if hasattr(lm, "model") else lm
    except Exception:
        pass
    # 2) Try vision-language models and extract the language sub-module
    try:
        from transformers import AutoModelForImageTextToText
        vl = AutoModelForImageTextToText.from_pretrained(model)
        for attr in ["language_model", "text_model", "model"]:
            if hasattr(vl, attr):
                lm = getattr(vl, attr)
                return lm.model if hasattr(lm, "model") else lm
        return vl
    except Exception as e:
        raise ValueError(f"Unknown model: {model}") from e


def update_self_attention_mask(llm_attention_mask: torch.LongTensor, dit_sequence_length: int, use_cache: bool, device, dtype):
    llm_sequence_length = llm_attention_mask.shape[1]
    sequence_length = dit_sequence_length + llm_sequence_length
    min_dtype = torch.finfo(dtype).min

    attention_mask = torch.full(
        (llm_attention_mask.shape[0], 1, sequence_length, sequence_length),
        fill_value=min_dtype,
        device=device,
        dtype=dtype,
    )

    attention_mask[:, :, :llm_sequence_length, :llm_sequence_length] = torch.triu(attention_mask[:, :, :llm_sequence_length, :llm_sequence_length], diagonal=1) # Causal mask for LLM
    attention_mask[:, :, llm_sequence_length:, :] = 0 # Bi-directional mask for DiT

    padding_mask = attention_mask[:, :, :, :llm_sequence_length] + repeat(llm_attention_mask, "b n -> b 1 1 n")
    padding_mask = padding_mask == 0
    attention_mask[:, :, :, :llm_sequence_length] = attention_mask[:, :, :, :llm_sequence_length].masked_fill(padding_mask, min_dtype)  # Padding mask for LLM
   
    if use_cache:
        attention_mask = attention_mask[:, :, llm_sequence_length:, :]

    return attention_mask


def update_cross_attention_mask(llm_attention_mask: torch.LongTensor, dit_sequence_length: int, joint: bool, device, dtype):
    min_dtype = torch.finfo(dtype).min

    if joint:
        llm_attention_mask = torch.cat([llm_attention_mask, torch.ones(llm_attention_mask.shape[0], dit_sequence_length, device=device, dtype=dtype)], dim=-1)

    cross_attention_mask = repeat(llm_attention_mask, "b n -> b 1 s n", s=dit_sequence_length).to(dtype=torch.bool)
    cross_attention_mask = torch.zeros_like(cross_attention_mask, device=device, dtype=dtype).masked_fill(cross_attention_mask.logical_not(), min_dtype)

    return cross_attention_mask


def unpatchify(dit_hidden_states: torch.FloatTensor, height: int, width: int, patch_size: int):
    # unpatchify
    output = rearrange(
        dit_hidden_states,
        "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
        h=height,
        w=width,
        p1=patch_size,
        p2=patch_size,
    )

    return output


def _patch_base_config_defaults(base_config):
    """Fill missing fields to support non-Gemma families (e.g., Qwen/Qwen-VL)."""
    # head_dim
    if not hasattr(base_config, "head_dim") and hasattr(base_config, "hidden_size") and hasattr(base_config, "num_attention_heads"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads
    # KV heads
    if not hasattr(base_config, "num_key_value_heads") and hasattr(base_config, "num_attention_heads"):
        base_config.num_key_value_heads = base_config.num_attention_heads
    # attention bias
    if not hasattr(base_config, "attention_bias"):
        base_config.attention_bias = False
    # rms norm eps
    if not hasattr(base_config, "rms_norm_eps"):
        base_config.rms_norm_eps = 1e-6
    # max position embeddings
    if not hasattr(base_config, "max_position_embeddings"):
        base_config.max_position_embeddings = 32768 if hasattr(base_config, "rope_theta") else 2048
    # rope theta
    if not hasattr(base_config, "rope_theta"):
        base_config.rope_theta = 10000.0
    # initializer range
    if not hasattr(base_config, "initializer_range"):
        base_config.initializer_range = 0.02
    return base_config


def _make_rope_config(base_config, use_base_rope: bool):
    """
    Create a temporary config for RoPE initialization.
    - use_base_rope=True  : reuse base_config max_position_embeddings and rope_theta (for FuseDiT with native LLM RoPE)
    - use_base_rope=False : fallback to theta=10000 and max_position_embeddings=4096
    """
    cfg = deepcopy(base_config)
    if not use_base_rope:
        cfg.max_position_embeddings = 4096
        cfg.rope_theta = 10000.0
    return cfg


class DiT(PreTrainedModel):
    """
    Diffusion Transformer Model.
    """

    config_class = DiTConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True  # * Make Transformers happy

    def __init__(self, config: DiTConfig):
        super().__init__(config)

        self.layers = nn.ModuleList(
            [DiTLayer(config) for _ in range(config.dit_num_hidden_layers)]
        )

        self.patch_embed = PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.dit_hidden_size,
            pos_embed_type=None if not config.pos_embed == "ape" else "sincos",
            pos_embed_max_size=config.pos_embed_max_size,
        )

        rope_cfg = _make_rope_config(config.base_config, use_base_rope=(config.model_type == "FuseDiT"))
        self.rotary_emb = GemmaRotaryEmbedding(rope_cfg)
        if config.pos_embed == "2d-rope":
            rope_half_cfg = deepcopy(rope_cfg)
            if hasattr(rope_half_cfg, "head_dim"):
                rope_half_cfg.head_dim = rope_half_cfg.head_dim // 2
            self.rotary_emb_half_module = GemmaRotaryEmbedding(rope_half_cfg)

        
        if config.timestep_conditioning is not None:
            self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder = nn.Sequential(
                nn.Linear(256, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        if config.timestep_conditioning == "adaln-single":
            self.t_block = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, 6 * config.dit_hidden_size),
            )
            self.scale_shift_table = nn.Parameter(torch.randn(2 * config.dit_hidden_size) / config.dit_hidden_size ** 0.5)

        if config.text_modulation_embeds_dim is not None:
            self.condition_embedder = nn.Sequential(
                RMSNorm(config.text_modulation_embeds_dim, eps=config.base_config.rms_norm_eps),
                nn.Linear(config.text_modulation_embeds_dim, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        if config.model_type == "DiT":
            self.context_embedder = nn.Sequential(
                RMSNorm(config.text_hidden_size, eps=config.base_config.rms_norm_eps),
                nn.Linear(config.text_hidden_size, config.text_hidden_size),
            )

        if config.timestep_conditioning == "adaln-zero":
            self.norm_out = AdaLayerNormOut(config)
        else:
            self.norm_out = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        self.proj_out = nn.Linear(
            config.dit_hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False

        self.initialize_weights()



    def rotary_emb_half(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        """
        Compute RoPE embeddings for half of the rotary dimension (used in 2d-rope).
        """
        return self.rotary_emb_half_module(hidden_states, position_ids)

    def prepare_hidden_states(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor, height: int, width: int):
        if not self.config.pos_embed == "ape":
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)

            if self.config.pos_embed == "m-rope" or self.config.pos_embed == "2d-rope":
                h_position_ids = torch.arange(height, device=hidden_states.device).repeat_interleave(width)
                h_position_ids = repeat(h_position_ids, "s -> b s", b=hidden_states.shape[0])
                w_position_ids = torch.arange(width, device=hidden_states.device).repeat(height)
                w_position_ids = repeat(w_position_ids, "s -> b s", b=hidden_states.shape[0])

                if self.config.pos_embed == "m-rope":
                    h_cos, h_sin = self.rotary_emb(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=0)
                    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.chunk(4, dim=-1))], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=0)
                    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.chunk(4, dim=-1))], dim=-1)
                elif self.config.pos_embed == "2d-rope":
                    h_cos, h_sin = self.rotary_emb_half(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb_half(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=2)
                    cos = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in cos.chunk(2, dim=-1)], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=2)
                    sin = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in sin.chunk(2, dim=-1)], dim=-1)

                pos_embed = (cos, sin)
            else:
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)

                pos_embed = self.rotary_emb(hidden_states, position_ids)
        else:
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)

            pos_embed = None

        return hidden_states, pos_embed


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        text_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = self.patch_embed(hidden_states)
        
        if self.config.timestep_conditioning is not None:
            timestep = self.time_proj(timestep).to(hidden_states.dtype)
            timestep = self.timestep_embedder(timestep)

            if self.config.timestep_conditioning == "adaln-single":
                temb = self.t_block(timestep)
            else:
                temb = timestep
        else:
            temb = torch.zeros_like(hidden_states[:, 0, :])

        hidden_states, pos_embed = self.prepare_hidden_states(hidden_states, temb, height, width)
        
        # Handle text features: optional normalized mean fusion (no trainable weights), only when multiple layers and use_norm_avg_fusion=True
        if text_hidden_states is not None:
            if isinstance(text_hidden_states, (list, tuple)):
                if self.config.use_norm_avg_fusion:
                    dtype = text_hidden_states[0].dtype
                    hidden_size = self.config.text_hidden_size or text_hidden_states[0].shape[-1]
                    normalized_hidden_states = [
                        F.layer_norm(h.float(), normalized_shape=(hidden_size,)) for h in text_hidden_states
                    ]
                    text_hidden_states = torch.mean(torch.stack(normalized_hidden_states, dim=0), dim=0).to(dtype)
                else:
                    raise ValueError(
                        "DiT received multiple text_hidden_states but use_norm_avg_fusion is False; "
                        "please enable use_norm_avg_fusion or pass a single tensor."
                    )
            # Force float32 for context_embedder (which contains Norm)
            else:
                dtype = text_hidden_states.dtype
                text_hidden_states = F.layer_norm(text_hidden_states, [text_hidden_states.shape[-1]]).to(dtype)
            text_hidden_states = self.context_embedder(text_hidden_states).to(dtype)

        cross_attention_mask = update_cross_attention_mask(
            attention_mask,
            hidden_states.shape[1],
            self.config.attention == "self",
            hidden_states.device,
            torch.float32 if ACCEL == "xla" else text_hidden_states.dtype,
        )

        for layer_index in range(self.config.dit_num_hidden_layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    self.layers[layer_index].__call__,
                    hidden_states,
                    temb,
                    pos_embed,
                    text_hidden_states,
                    cross_attention_mask,
                )

            else:
                hidden_states = self.layers[layer_index](hidden_states, temb, pos_embed, text_hidden_states, cross_attention_mask)

        if self.config.timestep_conditioning == "adaln-zero":
            hidden_states = self.norm_out(hidden_states, temb)
        else:
            # Force float32 for norm_out
            dtype = hidden_states.dtype
            hidden_states = self.norm_out(hidden_states.to(dtype=torch.float32)).to(dtype)
            
        if self.config.timestep_conditioning == "adaln-single":
            shift, scale = (self.scale_shift_table + repeat(timestep, "b d -> b (2 d)")).chunk(2, dim=1)
            hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out(hidden_states)

        output = unpatchify(hidden_states, height, width, patch_size)

        return output

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks and output layers:
        if self.config.timestep_conditioning == "adaln-zero":
            for layer in self.layers:
                nn.init.constant_(layer.input_layernorm.linear.weight, 0)
                nn.init.constant_(layer.input_layernorm.linear.bias, 0)

        if self.config.timestep_conditioning == "adaln-zero":
            nn.init.constant_(self.norm_out.linear.weight, 0)
            nn.init.constant_(self.norm_out.linear.bias, 0)
        elif self.config.timestep_conditioning == "adaln-single":
            nn.init.normal_(self.t_block[1].weight, std=self.config.base_config.initializer_range)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize timestep embedding MLP:
        if self.config.timestep_conditioning is not None:
            nn.init.normal_(self.timestep_embedder[0].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.timestep_embedder[2].weight, std=self.config.base_config.initializer_range)

        if self.config.text_modulation_embeds_dim is not None:
            # Initialize condition embedding MLP:
            nn.init.normal_(self.condition_embedder[1].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.condition_embedder[3].weight, std=self.config.base_config.initializer_range)
        if self.config.model_type == "DiT":
            # Initialize context embedding MLP:
            nn.init.normal_(self.context_embedder[1].weight, std=self.config.base_config.initializer_range)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = { "use_reentrant": True }

        gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)
        
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())


class FuseDiT(PreTrainedModel):
    """
    FuseDiT Model.
    """

    config_class = FuseDiTConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True  # * Make Transformers happy

    def __init__(self, config: FuseDiTConfig):
        super().__init__(config)

        if config.base_config.model_type == "gemma":
            self.llm = GemmaModel(config.base_config)
        elif config.base_config.model_type == "gemma2":
            self.llm = Gemma2Model(config.base_config)
        elif config.base_config.model_type == "qwen3_vl_text":
           
            self.llm = Qwen3VLTextModel(config.base_config)
            
        else:
            raise ValueError(f"Model type {config.base_config.model_type} not supported.")

        self.dit = DiT(config)

        self.num_key_value_groups = config.base_config.num_attention_heads // config.base_config.num_key_value_heads

        self.gradient_checkpointing = False

    def shared_transformer_layer(
        self,
        index: int,
        dit_hidden_states: torch.FloatTensor,
        llm_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        pos_embed: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        cross_attention_mask: Optional[torch.FloatTensor],
        past_key_values: Optional[DynamicCache] = None,
    ):
        """
        Joint Transformer Layer.
        """

        if self.config.timestep_conditioning == "adaln-zero":
            (
                norm_dit_hidden_states,
                dit_gate_msa,
                dit_shift_mlp,
                dit_scale_mlp,
                dit_gate_mlp,
            ) = self.dit.layers[index - self.config.initial_layers].input_layernorm(dit_hidden_states, emb=temb)
            dit_scale_msa = dit_shift_msa = torch.zeros_like(temb) # Dummy values.
        else:
            norm_dit_hidden_states = self.dit.layers[index - self.config.initial_layers].input_layernorm(dit_hidden_states)
            if self.config.timestep_conditioning == "adaln-single":
                (
                    dit_shift_msa,
                    dit_scale_msa,
                    dit_gate_msa,
                    dit_shift_mlp,
                    dit_scale_mlp,
                    dit_gate_mlp
                ) = (self.dit.layers[index - self.config.initial_layers].scale_shift_table + temb).chunk(6, dim=1)
            else:
                dit_shift_msa = dit_scale_msa = dit_shift_mlp = dit_scale_mlp = torch.zeros_like(temb)
                dit_gate_msa = dit_gate_mlp = torch.ones_like(temb)

        norm_dit_hidden_states = norm_dit_hidden_states * (1 + dit_scale_msa[:, None]) + dit_shift_msa[:, None]
        if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
            norm_llm_hidden_states = self.llm.layers[index].input_layernorm(llm_hidden_states)

        ########## Self Attention Begins ##########

        dit_query_states = self.dit.layers[index - self.config.initial_layers].self_attn.q_proj(norm_dit_hidden_states)
        dit_key_states = self.dit.layers[index - self.config.initial_layers].self_attn.k_proj(norm_dit_hidden_states)
        dit_value_states = self.dit.layers[index - self.config.initial_layers].self_attn.v_proj(norm_dit_hidden_states)

        dit_query_states = rearrange(dit_query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
        dit_key_states = rearrange(dit_key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
        dit_value_states = rearrange(dit_value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
            if self.config.base_config.model_type == "qwen3_vl_text":
                input_shape = norm_llm_hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.config.base_config.head_dim)

                llm_query_states = self.llm.layers[index].self_attn.q_norm(
                    self.llm.layers[index].self_attn.q_proj(norm_llm_hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                llm_key_states = self.llm.layers[index].self_attn.k_norm(
                    self.llm.layers[index].self_attn.k_proj(norm_llm_hidden_states).view(hidden_shape)
                ).transpose(1, 2)
                llm_value_states = self.llm.layers[index].self_attn.v_proj(norm_llm_hidden_states).view(hidden_shape).transpose(1, 2)
            else:
                llm_query_states = self.llm.layers[index].self_attn.q_proj(norm_llm_hidden_states)
                llm_key_states = self.llm.layers[index].self_attn.k_proj(norm_llm_hidden_states)
                llm_value_states = self.llm.layers[index].self_attn.v_proj(norm_llm_hidden_states)
            
                llm_query_states = rearrange(llm_query_states, "b n (h d) -> b h n d", h=self.config.base_config.num_attention_heads)
                llm_key_states = rearrange(llm_key_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)
                llm_value_states = rearrange(llm_value_states, "b n (h d) -> b h n d", h=self.config.base_config.num_key_value_heads)

        if self.config.qk_norm:
            dit_query_states = self.dit.layers[index - self.config.initial_layers].self_attn.q_norm(dit_query_states)
            dit_key_states = self.dit.layers[index - self.config.initial_layers].self_attn.k_norm(dit_key_states)

        if self.config.attention == "self":
            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                if self.config.pos_embed == "ape": # RoPE only for LLM, APE for DiT
                    cos, sin = pos_embed
                    llm_query_states, llm_key_states = apply_rotary_pos_emb(llm_query_states, llm_key_states, cos, sin)

                query_states = torch.cat([llm_query_states, dit_query_states], dim=2)
                key_states = torch.cat([llm_key_states, dit_key_states], dim=2)
                value_states = torch.cat([llm_value_states, dit_value_states], dim=2)

                if not self.config.pos_embed == "ape": # RoPE for both LLM and DiT
                    cos, sin = pos_embed
                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    past_key_values.update(key_states[:, :, :-dit_hidden_states.shape[1]], value_states[:, :, :-dit_hidden_states.shape[1]], index - self.config.initial_layers)
            else:
                if not self.config.pos_embed == "ape": # RoPE for both LLM and DiT
                    cos, sin = pos_embed
                    dit_query_states, dit_key_states = apply_rotary_pos_emb(dit_query_states, dit_key_states, cos, sin)

                query_states = dit_query_states
                key_states = torch.cat([past_key_values.key_cache[index - self.config.initial_layers], dit_key_states], dim=2)
                value_states = torch.cat([past_key_values.value_cache[index - self.config.initial_layers], dit_value_states], dim=2)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, is_causal=False)

            attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

            dit_attn_output = attn_output[:, -dit_hidden_states.shape[1]:]
            dit_attn_output = self.dit.layers[index - self.config.initial_layers].self_attn.o_proj(dit_attn_output)
            if self.config.sandwich_norm:
                dit_attn_output = self.dit.layers[index - self.config.initial_layers].post_attention_layernorm(dit_attn_output)
            dit_hidden_states = dit_hidden_states + dit_gate_msa.unsqueeze(1) * dit_attn_output

            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                llm_attn_output = attn_output[:, :-dit_hidden_states.shape[1]]
                llm_attn_output = self.llm.layers[index].self_attn.o_proj(llm_attn_output)
                if self.config.base_config.model_type == "gemma2":
                    llm_attn_output = self.llm.layers[index].post_attention_layernorm(llm_attn_output)
                llm_hidden_states = llm_hidden_states + llm_attn_output
        elif self.config.attention == "cross":
            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                if past_key_values is not None:
                    past_key_values.update(llm_key_states, llm_value_states, index - self.config.initial_layers)

                if self.config.pos_embed == "ape": # RoPE only for LLM, APE for DiT
                    cos, sin = pos_embed
                    rope_llm_query_states, rope_llm_key_states = apply_rotary_pos_emb(llm_query_states, llm_key_states, cos, sin)
                else:
                    position_ids = torch.arange(
                        llm_hidden_states.shape[1], device=llm_hidden_states.device
                    ).unsqueeze(0)
                    llm_pos_embed = self.dit.rotary_emb(llm_hidden_states, position_ids)
                    cos, sin = llm_pos_embed
                    rope_llm_query_states, rope_llm_key_states = apply_rotary_pos_emb(llm_query_states, llm_key_states, cos, sin)

                rope_llm_key_states = repeat_kv(rope_llm_key_states, self.num_key_value_groups)
                llm_key_states = repeat_kv(llm_key_states, self.num_key_value_groups)
                llm_value_states = repeat_kv(llm_value_states, self.num_key_value_groups)

                llm_attn_output = F.scaled_dot_product_attention(
                    rope_llm_query_states,
                    rope_llm_key_states,
                    llm_value_states,
                    attn_mask=attention_mask,
                    is_causal=False
                )

                llm_attn_output = rearrange(llm_attn_output, "b h n d -> b n (h d)")

                llm_attn_output = self.llm.layers[index].self_attn.o_proj(llm_attn_output)

                if self.config.base_config.model_type == "gemma2":
                    llm_attn_output = self.llm.layers[index].post_attention_layernorm(llm_attn_output)
            else:
                llm_key_states = past_key_values.key_cache[index - self.config.initial_layers]
                llm_value_states = past_key_values.value_cache[index - self.config.initial_layers]

            if not self.config.pos_embed == "ape": # RoPE for both LLM and DiT
                cos, sin = pos_embed
                dit_query_states, dit_key_states = apply_rotary_pos_emb(dit_query_states, dit_key_states, cos, sin)

            dit_key_states = repeat_kv(dit_key_states, self.num_key_value_groups)
            dit_value_states = repeat_kv(dit_value_states, self.num_key_value_groups)

            dit_attn_output = F.scaled_dot_product_attention(
                dit_query_states,
                dit_key_states,
                dit_value_states,
                attn_mask=None,
                is_causal=False
            )

            dit_attn_output = rearrange(dit_attn_output, "b h n d -> b n (h d)")

            dit_attn_output = self.dit.layers[index - self.config.initial_layers].self_attn.o_proj(dit_attn_output)

            if self.config.sandwich_norm:
                dit_attn_output = self.dit.layers[index - self.config.initial_layers].post_attention_layernorm(dit_attn_output)

        ########## Self Attention Ends ##########

        ########## Cross Attention Begins ##########

            dit_hidden_states = dit_hidden_states + dit_gate_msa.unsqueeze(1) * dit_attn_output

            dit_cross_attn_output = self.dit.layers[index - self.config.initial_layers].cross_attn(
                dit_hidden_states,
                key_states=llm_key_states.to(dit_hidden_states.dtype),
                value_states=llm_value_states.to(dit_hidden_states.dtype),
                attention_mask=cross_attention_mask
            )

            if self.config.sandwich_norm:
                dit_cross_attn_output = self.dit.layers[index - self.config.initial_layers].post_cross_attention_layernorm(dit_cross_attn_output)

            dit_hidden_states = dit_hidden_states + dit_cross_attn_output

            if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
                llm_hidden_states = llm_hidden_states + llm_attn_output

        ########## Cross Attention Ends ##########

        else:
            raise ValueError(f"Unknown attention type: {self.config.attention}")

        ########## Feedforward Begins ##########
        
        if self.config.sandwich_norm:
            norm_dit_hidden_states = self.dit.layers[index - self.config.initial_layers].pre_feedforward_layernorm(dit_hidden_states)
        else:
            norm_dit_hidden_states = self.dit.layers[index - self.config.initial_layers].post_attention_layernorm(dit_hidden_states)
        norm_dit_hidden_states = norm_dit_hidden_states * (1 + dit_scale_mlp[:, None]) + dit_shift_mlp[:, None]
        dit_mlp_output = self.dit.layers[index - self.config.initial_layers].mlp(norm_dit_hidden_states)
        if self.config.sandwich_norm:
            dit_mlp_output = self.dit.layers[index - self.config.initial_layers].post_feedforward_layernorm(dit_mlp_output)
        dit_hidden_states = dit_hidden_states + dit_gate_mlp.unsqueeze(1) * dit_mlp_output

        if past_key_values is None or len(past_key_values.key_cache) < self.config.base_config.num_hidden_layers:
            if self.config.base_config.model_type == "qwen3_vl_text":
                norm_llm_hidden_states = self.llm.layers[index].post_attention_layernorm(llm_hidden_states)
                llm_mlp_output = self.llm.layers[index].mlp(norm_llm_hidden_states)
            else:
                if self.config.base_config.model_type == "gemma2":
                    norm_llm_hidden_states = self.llm.layers[index].pre_feedforward_layernorm(llm_hidden_states)
                else:
                    norm_llm_hidden_states = self.llm.layers[index].post_attention_layernorm(llm_hidden_states)
                llm_mlp_output = self.llm.layers[index].mlp(norm_llm_hidden_states)
                if self.config.base_config.model_type == "gemma2":
                    llm_mlp_output = self.llm.layers[index].post_feedforward_layernorm(llm_mlp_output)
            llm_hidden_states = llm_hidden_states + llm_mlp_output

        return dit_hidden_states, llm_hidden_states
    

    def prepare_hidden_states(
        self,
        llm_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        dit_hidden_states: torch.FloatTensor,
        llm_attention_mask: torch.LongTensor,
        height: int,
        width: int,
        use_cache: bool,
    ):
        dit_sequence_length = dit_hidden_states.shape[1]
        llm_sequence_length = llm_hidden_states.shape[1]
        
        if not self.config.pos_embed == "ape":
            if self.config.timestep_conditioning == "addition":
                dit_hidden_states += temb.unsqueeze(1)
            
            if self.config.pos_embed == "m-rope" or self.config.pos_embed == "2d-rope":
                h_position_ids = torch.arange(height, device=llm_hidden_states.device).repeat_interleave(width)
                h_position_ids = repeat(h_position_ids, "s -> b s", b=llm_hidden_states.shape[0])
                w_position_ids = torch.arange(width, device=llm_hidden_states.device).repeat(height)
                w_position_ids = repeat(w_position_ids, "s -> b s", b=llm_hidden_states.shape[0])

                if self.config.attention == "self" and not use_cache:
                    padding = (1 - llm_attention_mask).sum(-1, keepdim=True)
                    h_position_ids = h_position_ids + llm_sequence_length - padding # Remove padding tokens
                    w_position_ids = w_position_ids + llm_sequence_length - padding # Remove padding tokens

                if self.config.pos_embed == "m-rope":
                    if self.config.attention == "self" and not use_cache:                        
                        llm_position_ids = repeat(torch.arange(llm_sequence_length, device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                        h_position_ids = torch.cat([llm_position_ids, h_position_ids], dim=1)
                        w_position_ids = torch.cat([llm_position_ids, w_position_ids], dim=1)

                    h_cos, h_sin = self.dit.rotary_emb(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), h_position_ids)
                    w_cos, w_sin = self.dit.rotary_emb(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=0)
                    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.chunk(4, dim=-1))], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=0)
                    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.chunk(4, dim=-1))], dim=-1)

                    pos_embed = (cos, sin)

                elif self.config.pos_embed == "2d-rope":
                    # Use image token length; feature dim fixed to head_dim to align cos/sin with LLM
                    rope_hidden = llm_hidden_states.new_zeros(
                        dit_hidden_states.shape[0],
                        dit_hidden_states.shape[1],
                        self.config.base_config.head_dim,
                    )
                    h_cos, h_sin = self.dit.rotary_emb_half(rope_hidden, h_position_ids)
                    w_cos, w_sin = self.dit.rotary_emb_half(rope_hidden, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=2)
                    cos = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in cos.chunk(2, dim=-1)], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=2)
                    sin = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in sin.chunk(2, dim=-1)], dim=-1)

                    if self.config.attention == "self" and not use_cache:
                        llm_position_ids = repeat(torch.arange(llm_sequence_length, device=llm_hidden_states.device), "s -> b s", b=llm_hidden_states.shape[0])
                        llm_cos, llm_sin = self.dit.rotary_emb(llm_hidden_states, llm_position_ids)

                        cos = torch.cat([llm_cos, cos], dim=1)
                        sin = torch.cat([llm_sin, sin], dim=1)

                    pos_embed = (cos, sin)
            else:
                if self.config.attention == "self" or not use_cache:
                    position_ids = torch.arange(
                        llm_sequence_length + dit_sequence_length, device=llm_hidden_states.device
                    ).unsqueeze(0).expand(llm_hidden_states.shape[0], -1).contiguous()
                    position_ids[:, llm_sequence_length:] -= (1 - llm_attention_mask).sum(-1, keepdim=True) # Remove padding tokens
                else:
                    position_ids = torch.arange(
                        dit_sequence_length, device=dit_hidden_states.device
                    ).unsqueeze(0).expand(dit_hidden_states.shape[0], -1).contiguous()

                pos_embed = self.dit.rotary_emb(torch.cat([llm_hidden_states, dit_hidden_states], dim=1), position_ids)
        else:
            if self.config.timestep_conditioning == "addition":
                dit_hidden_states += temb.unsqueeze(1)
                
            position_ids = torch.arange(
                llm_hidden_states.shape[1], device=llm_hidden_states.device
            ).unsqueeze(0)
            pos_embed = self.dit.rotary_emb(llm_hidden_states, position_ids)

        return dit_hidden_states, pos_embed

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        text_modulation_embeds: Optional[torch.FloatTensor] = None,

        use_cache: bool = False,
        past_key_values: Optional[DynamicCache] = None,
    ):
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        dit_hidden_states = self.dit.patch_embed(hidden_states)
        if past_key_values is None:
            if use_cache:
                past_key_values = DynamicCache()

            llm_hidden_states = self.llm.embed_tokens(input_ids)
            if "gemma" in self.config.base_config.model_type:
                llm_hidden_states *= torch.tensor(self.config.base_config.hidden_size ** 0.5, dtype=self.llm.dtype)
        else:
            llm_hidden_states = torch.zeros(hidden_states.shape[0], past_key_values.get_seq_length(), self.config.base_config.hidden_size, device=hidden_states.device)

        if self.config.timestep_conditioning is not None:
            timestep = self.dit.time_proj(timestep).to(dit_hidden_states.dtype)
            timestep = self.dit.timestep_embedder(timestep)

            if self.config.text_modulation_embeds_dim is not None:
                timestep = timestep + self.dit.condition_embedder(text_modulation_embeds)

            if self.config.timestep_conditioning == "adaln-single":
                temb = self.dit.t_block(timestep)
            else:
                temb = timestep
        else:
            temb = torch.zeros_like(dit_hidden_states[:, 0, :])

        dit_hidden_states, pos_embed = self.prepare_hidden_states(
            llm_hidden_states,
            temb,
            dit_hidden_states,
            attention_mask,
            height,
            width,
            use_cache and past_key_values.get_seq_length() > 0,
        )

        self_attention_mask = update_self_attention_mask(
            attention_mask,
            dit_hidden_states.shape[1] if self.config.attention == "self" else 0,
            use_cache and past_key_values.get_seq_length() > 0,
            llm_hidden_states.device,
            torch.float32 if ACCEL == "xla" else llm_hidden_states.dtype,
        )

        if self.config.attention == "cross":
            cross_attention_mask = update_cross_attention_mask(
                attention_mask,
                dit_hidden_states.shape[1],
                False,
                llm_hidden_states.device,
                torch.float32 if ACCEL == "xla" else llm_hidden_states.dtype,
            )
        else:
            cross_attention_mask = None

        for layer_index in range(self.config.initial_layers):
            if self.gradient_checkpointing and self.training:
                llm_pos_ids = repeat(
                    torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device),
                    "s -> b s",
                    b=llm_hidden_states.shape[0],
                )
                llm_pos_embed = self.dit.rotary_emb(llm_hidden_states, llm_pos_ids)
                llm_hidden_states = self._gradient_checkpointing_func(
                    self.llm.layers[layer_index].__call__,
                    hidden_states=llm_hidden_states,
                    position_embeddings=llm_pos_embed,
                    attention_mask=self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                    position_ids=llm_pos_ids,
                )  # type: ignore

            else:
                llm_pos_ids = repeat(
                    torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device),
                    "s -> b s",
                    b=llm_hidden_states.shape[0],
                )
                llm_pos_embed = self.dit.rotary_emb(llm_hidden_states, llm_pos_ids)
                #print(f"llm_pos_embed shape: {llm_pos_embed[0].shape}")
                #print(f"llm_hidden_states shape before layer {layer_index}: {llm_hidden_states.shape}")
                llm_hidden_states = self.llm.layers[layer_index](
                    hidden_states=llm_hidden_states,
                    position_embeddings=llm_pos_embed,
                    attention_mask=self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                    position_ids=llm_pos_ids,
                )

        for layer_index in range(self.config.initial_layers, self.config.dit_num_hidden_layers+self.config.initial_layers):
            if self.gradient_checkpointing and self.training:
                if self.config.shared_attention_layers == "all" or layer_index in self.config.shared_attention_layers:
                    dit_hidden_states, llm_hidden_states = (
                        self._gradient_checkpointing_func(
                            self.shared_transformer_layer,
                            layer_index,
                            dit_hidden_states,
                            llm_hidden_states,
                            temb,
                            pos_embed,
                            self_attention_mask,
                            cross_attention_mask,
                            past_key_values,
                        )
                    )
                else:
                    dit_hidden_states = self._gradient_checkpointing_func(
                        self.dit.layers[layer_index].__call__,
                        dit_hidden_states,
                        temb,
                        None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                    )
                    llm_pos_ids = repeat(
                        torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device),
                        "s -> b s",
                        b=llm_hidden_states.shape[0],
                    )
                    llm_pos_embed = self.dit.rotary_emb(llm_hidden_states, llm_pos_ids)
                    llm_hidden_states = self._gradient_checkpointing_func(
                        self.llm.layers[layer_index].__call__,
                        hidden_states=llm_hidden_states,
                        position_embeddings=llm_pos_embed,
                        attention_mask=self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                        position_ids=llm_pos_ids,
                    ) 

            else:
                if self.config.shared_attention_layers == "all" or layer_index in self.config.shared_attention_layers:
                    dit_hidden_states, llm_hidden_states = self.shared_transformer_layer(
                        layer_index,
                        dit_hidden_states,
                        llm_hidden_states,
                        temb,
                        pos_embed,
                        self_attention_mask,
                        cross_attention_mask,
                        past_key_values,
                    )
                else:
                    dit_hidden_states = self.dit.layers[layer_index](
                        dit_hidden_states,
                        temb,
                        None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                    )
                    llm_pos_ids = repeat(
                        torch.arange(llm_hidden_states.shape[1], device=llm_hidden_states.device),
                        "s -> b s",
                        b=llm_hidden_states.shape[0],
                    )
                    llm_pos_embed = self.dit.rotary_emb(llm_hidden_states, llm_pos_ids)
                    llm_hidden_states = self.llm.layers[layer_index](
                        hidden_states=llm_hidden_states,
                        position_embeddings=llm_pos_embed,
                        attention_mask=self_attention_mask[:, :, :llm_hidden_states.shape[1], :llm_hidden_states.shape[1]],
                        position_ids=llm_pos_ids,
                    )

        for layer_index in range(self.config.base_config.num_hidden_layers - self.config.initial_layers, self.config.dit_num_hidden_layers):
            if self.gradient_checkpointing and self.training:
                dit_hidden_states = self._gradient_checkpointing_func(
                    self.dit.layers[layer_index].__call__,
                    dit_hidden_states,
                    temb,
                    None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                )

            else:
                dit_hidden_states = self.dit.layers[layer_index](
                    dit_hidden_states,
                    temb,
                    None if pos_embed is None else (pos_embed[0][:, llm_hidden_states.shape[1]:], pos_embed[1][:, llm_hidden_states.shape[1]:]),
                )

        if self.config.timestep_conditioning == "adaln-zero":
            dit_hidden_states = self.dit.norm_out(dit_hidden_states, temb)
        else:
            dit_hidden_states = self.dit.norm_out(dit_hidden_states)
        if self.config.timestep_conditioning == "adaln-single":
            shift, scale = (self.dit.scale_shift_table + repeat(timestep, "b d -> b (2 d)")).chunk(2, dim=1)
            dit_hidden_states = dit_hidden_states * (1 + scale[:, None]) + shift[:, None]
        dit_hidden_states = self.dit.proj_out(dit_hidden_states)

        output = (
            unpatchify(dit_hidden_states, height, width, patch_size),
            past_key_values,
        )

        return output
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = { "use_reentrant": True }

        gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)
        
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def trainable_parameters(self): # * Make SPMD CheckpointManager happy
        return (
            [param for param in self.dit.parameters() if param.requires_grad]
            + [param for param in self.llm.embed_tokens.parameters() if param.requires_grad]
            + [param for param in self.llm.layers[:-1].parameters() if param.requires_grad]
            + [param for name, param in self.llm.layers[-1].named_parameters()
                if not "o_proj" in name
                and not "post_attention_layernorm" in name
                and not "pre_feedforward_layernorm" in name
                and not "post_feedforward_layernorm" in name
                and not "mlp" in name
                and param.requires_grad
            ]
        )

class AdaFuseDiT(PreTrainedModel):
    """
    Adaptive Fuse Diffusion Transformer Model.
    
    AdaFuseDiT augments the standard DiT with timestep-aware feature weighting.
    It dynamically fuses features from multiple text encoder layers across diffusion timesteps,
    enabling more flexible text-image conditioning.

    Supported fusion modes (controlled by config):
    1. Global fixed weights: use_timestep_adaptive_fusion=False, use_layer_wise_fusion=False
    2. Global timestep-adaptive weights: use_timestep_adaptive_fusion=True, use_layer_wise_fusion=False
    3. Per-layer fixed weights: use_timestep_adaptive_fusion=False, use_layer_wise_fusion=True
    4. Per-layer timestep-adaptive weights: use_timestep_adaptive_fusion=True, use_layer_wise_fusion=True
    """

    config_class = DiTConfig
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def __init__(self, config: DiTConfig):
        super().__init__(config)

        # Standard DiT layers
        self.layers = nn.ModuleList(
            [DiTLayer(config) for _ in range(config.dit_num_hidden_layers)]
        )

        self.patch_embed = PatchEmbed(
            height=config.sample_size,
            width=config.sample_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.dit_hidden_size,
            pos_embed_type=None if not config.pos_embed == "ape" else "sincos",
            pos_embed_max_size=config.pos_embed_max_size,
        )

        
        import copy
        gemma_config=copy.deepcopy(config.base_config)
        gemma_config.rope_theta = 1e5
        gemma_config.rope_type="default"
        gemma_config.max_position_embeddings = 2048
        self.rotary_emb = GemmaRotaryEmbedding(
            gemma_config
        )
        if config.pos_embed == "2d-rope":
            gemma_config.head_dim = gemma_config.head_dim // 2
            self.rotary_emb_half_module = GemmaRotaryEmbedding(
                gemma_config
            )
        
        
        if config.timestep_conditioning is not None:
            self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.timestep_embedder = nn.Sequential(
                nn.Linear(256, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        if config.timestep_conditioning == "adaln-single":
            self.t_block = nn.Sequential(
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, 6 * config.dit_hidden_size),
            )
            self.scale_shift_table = nn.Parameter(torch.randn(2 * config.dit_hidden_size) / config.dit_hidden_size ** 0.5)

        if config.text_modulation_embeds_dim is not None:
            self.condition_embedder = nn.Sequential(
                RMSNorm(config.text_modulation_embeds_dim, eps=config.base_config.rms_norm_eps),
                nn.Linear(config.text_modulation_embeds_dim, config.dit_hidden_size),
                nn.SiLU(),
                nn.Linear(config.dit_hidden_size, config.dit_hidden_size),
            )
        # === AdaFuseDiT core: build text fusion modules per config ===
        if config.text_hidden_states_num > 1:
            if config.use_layer_wise_fusion:
                # Per-layer fusion
                if config.use_timestep_adaptive_fusion:
                    # Mode 4: per-layer TimestepWiseFeatureWeighting
                    self.text_fusion_modules = nn.ModuleList([
                        TimestepWiseFeatureWeighting(
                            num_layers=config.text_hidden_states_num,
                            time_embed_dim=config.adaptive_fusion_time_embed_dim,
                            feature_dim=config.text_hidden_size
                        )
                        for _ in range(config.dit_num_hidden_layers)
                    ])
                else:
                    # Mode 3: per-layer learnable weights
                    # Zero-init so softmax(zeros) is uniform
                    self.text_fusion_weights = nn.ParameterList([
                        nn.Parameter(torch.zeros(config.text_hidden_states_num))
                        for _ in range(config.dit_num_hidden_layers)
                    ])
            else:
                # Global fusion shared across layers
                if config.use_timestep_adaptive_fusion:
                    # Mode 2: global TimestepWiseFeatureWeighting
                    self.text_fusion_module = TimestepWiseFeatureWeighting(
                        num_layers=config.text_hidden_states_num,
                        time_embed_dim=config.adaptive_fusion_time_embed_dim,
                        feature_dim=config.text_hidden_size
                    )
                else:
                    # Mode 1: global learnable weights
                    # Zero-init so softmax(zeros) is uniform
                    self.text_fusion_weight = nn.Parameter(
                        torch.zeros(config.text_hidden_states_num)
                    )
        
        
        # Map fused text features to DiT hidden dimension
        self.context_embedder = nn.Sequential(
            RMSNorm(config.text_hidden_size, eps=config.base_config.rms_norm_eps),
            nn.Linear(config.text_hidden_size, config.text_hidden_size),
        )

        if config.timestep_conditioning == "adaln-zero":
            self.norm_out = AdaLayerNormOut(config)
        else:
            self.norm_out = RMSNorm(config.dit_hidden_size, eps=config.base_config.rms_norm_eps)
        self.proj_out = nn.Linear(
            config.dit_hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )

        self.gradient_checkpointing = False

        self.initialize_weights()
    def rotary_emb_half(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        """
        Compute RoPE embeddings for half of the rotary dimension (used in 2d-rope).
        """
        return self.rotary_emb_half_module(hidden_states, position_ids)
    def _fuse_text_features(
        self,
        text_hidden_states: Optional[list],
        timestep: torch.Tensor,
    ):
        """
        Vectorized text feature fusion.
        Returns:
        - fused_text_global: globally fused features (non layer-wise mode)
        - fused_text_per_layer: list of per-layer fused features (layer-wise mode)
        """
        if text_hidden_states is None:
            return None, None

        if isinstance(text_hidden_states, (list, tuple)):
            stacked = torch.stack(text_hidden_states, dim=0).float()  # (L, B, S, C)
            text_dtype = text_hidden_states[0].dtype
        else:
            stacked = text_hidden_states.unsqueeze(0).float()  # (1, B, S, C)
            text_dtype = text_hidden_states.dtype

        stacked = F.layer_norm(stacked, (stacked.shape[-1],))

        fused_text_global = None
        fused_text_per_layer = None

        if not self.config.use_layer_wise_fusion:
            if self.config.use_timestep_adaptive_fusion:
                normalized_timestep = timestep.float() / 1000.0
                t_embed = self.text_fusion_module._time_embedding(normalized_timestep)
                weight_logits = self.text_fusion_module.weight_generator(t_embed.to(dtype=stacked.dtype))
                weights = F.softmax(weight_logits, dim=-1)  # (b, L)
                fused_text_global = torch.einsum("bl,lbsc->bsc", weights, stacked)
            else:
                weights = F.softmax(self.text_fusion_weight.float(), dim=0)  # (L,)
                fused_text_global = torch.einsum("l,lbsc->bsc", weights, stacked)

            fused_text_global = fused_text_global.to(dtype=text_dtype)
            fused_text_global = self.context_embedder(fused_text_global).to(fused_text_global.dtype)
        else:
            fused_text_per_layer = []
            if self.config.use_timestep_adaptive_fusion:
                normalized_timestep = timestep.float() / 1000.0
                # Share timestep embeddings to compute all layer weights at once
                t_embed = self.text_fusion_modules[0]._time_embedding(normalized_timestep)
                weight_logits = torch.stack(
                    [module.weight_generator(t_embed.to(dtype=stacked.dtype)) for module in self.text_fusion_modules],
                    dim=0,
                )  # (K, B, L)
                weights = F.softmax(weight_logits, dim=-1)
                fused_stack = torch.einsum("kbl,lbsc->kbsc", weights, stacked)  # (K, B, S, C)
                for layer_idx in range(self.config.dit_num_hidden_layers):
                    fused_layer = fused_stack[layer_idx].to(dtype=text_dtype)
                    fused_text_per_layer.append(self.context_embedder(fused_layer).to(fused_layer.dtype))
            else:
                weight_stack = torch.stack([F.softmax(w.float(), dim=0) for w in self.text_fusion_weights], dim=0)  # (K, L)
                fused_stack = torch.einsum("kl,lbsc->kbsc", weight_stack, stacked)
                for layer_idx in range(self.config.dit_num_hidden_layers):
                    fused_layer = fused_stack[layer_idx].to(dtype=text_dtype)
                    fused_text_per_layer.append(self.context_embedder(fused_layer).to(fused_layer.dtype))

        return fused_text_global, fused_text_per_layer
    def prepare_hidden_states(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor, height: int, width: int):
        if not self.config.pos_embed == "ape":
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)

            if self.config.pos_embed == "m-rope" or self.config.pos_embed == "2d-rope":
                h_position_ids = torch.arange(height, device=hidden_states.device).repeat_interleave(width)
                h_position_ids = repeat(h_position_ids, "s -> b s", b=hidden_states.shape[0])
                w_position_ids = torch.arange(width, device=hidden_states.device).repeat(height)
                w_position_ids = repeat(w_position_ids, "s -> b s", b=hidden_states.shape[0])

                if self.config.pos_embed == "m-rope":
                    h_cos, h_sin = self.rotary_emb(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=0)
                    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.chunk(4, dim=-1))], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=0)
                    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.chunk(4, dim=-1))], dim=-1)
                elif self.config.pos_embed == "2d-rope":
                    h_cos, h_sin = self.rotary_emb_half(hidden_states, h_position_ids)
                    w_cos, w_sin = self.rotary_emb_half(hidden_states, w_position_ids)
                    cos = torch.stack([h_cos, w_cos], dim=2)
                    cos = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in cos.chunk(2, dim=-1)], dim=-1)
                    sin = torch.stack([h_sin, w_sin], dim=2)
                    sin = torch.cat([rearrange(m, "b s n d -> b s (n d)") for m in sin.chunk(2, dim=-1)], dim=-1)

                pos_embed = (cos, sin)
            else:
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                pos_embed = self.rotary_emb(hidden_states, position_ids)
        else:
            if self.config.timestep_conditioning == "addition":
                hidden_states += temb.unsqueeze(1)
            pos_embed = None

        return hidden_states, pos_embed

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.LongTensor,
        text_hidden_states: Optional[list] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass.

        Args:
            hidden_states: image hidden states, shape (batch_size, channels, height, width)
            timestep: diffusion timesteps, shape (batch_size,)
            text_hidden_states: text hidden states
                - if text_hidden_states_num > 1: list of tensors, each (batch_size, seq_len, hidden_size)
                - else: single tensor, shape (batch_size, seq_len, hidden_size)
            attention_mask: attention mask, shape (batch_size, seq_len)
        """
        height, width = hidden_states.shape[-2:]
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = self.patch_embed(hidden_states)
        


        if self.config.timestep_conditioning is not None:
            timestep_embed = self.time_proj(timestep).to(hidden_states.dtype)
            timestep_embed = self.timestep_embedder(timestep_embed)

            if self.config.timestep_conditioning == "adaln-single":
                temb = self.t_block(timestep_embed)
            else:
                temb = timestep_embed
        else:
            temb = torch.zeros_like(hidden_states[:, 0, :])

        hidden_states, pos_embed = self.prepare_hidden_states(hidden_states, temb, height, width)

        #if torch.isnan(hidden_states).any():
        #    raise ValueError("Prepared hidden_states contains NaN values.")
        cross_attention_mask = update_cross_attention_mask(
            attention_mask,
            hidden_states.shape[1],
            self.config.attention == "self",
            hidden_states.device,
            torch.float32 if ACCEL == "xla" else hidden_states.dtype,
        )

        fused_text_global, fused_text_per_layer = self._fuse_text_features(
            text_hidden_states,
            timestep,
        )

        for layer_idx in range(self.config.dit_num_hidden_layers):
            if fused_text_per_layer is not None:
                fused_text = fused_text_per_layer[layer_idx]
            else:
                fused_text = fused_text_global

            # Execute DiT layer
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    self.layers[layer_idx].__call__,
                    hidden_states,
                    temb,
                    pos_embed,
                    fused_text,
                    cross_attention_mask,
                )
            else:
                hidden_states = self.layers[layer_idx](
                    hidden_states,
                    temb,
                    pos_embed,
                    fused_text,
                    cross_attention_mask
                )
            #(hidden_states).any():
            #    raise ValueError(f"Hidden states after layer {layer_idx} contains NaN values.")
        if self.config.timestep_conditioning == "adaln-zero":
            hidden_states = self.norm_out(hidden_states, temb)
        else:
            # Force float32 for norm_out
            dtype = hidden_states.dtype
            hidden_states = self.norm_out(hidden_states.to(dtype=torch.float32)).to(dtype)
        #if torch.isnan(hidden_states).any():
        #   raise ValueError("Hidden states after norm_out contains NaN values.")
        if self.config.timestep_conditioning == "adaln-single":
            shift, scale = (self.scale_shift_table + repeat(timestep_embed, "b d -> b (2 d)")).chunk(2, dim=1)
            hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out(hidden_states)

        #if torch.isnan(hidden_states).any():
        #    raise ValueError("Hidden states after proj_out contains NaN values.")
        output = unpatchify(hidden_states, height, width, patch_size)
        #if torch.isnan(output).any():
        #    raise ValueError("Output contains NaN values.")
        return output

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.config.timestep_conditioning == "adaln-zero":
            for layer in self.layers:
                nn.init.constant_(layer.input_layernorm.linear.weight, 0)
                nn.init.constant_(layer.input_layernorm.linear.bias, 0)

        if self.config.timestep_conditioning == "adaln-zero":
            nn.init.constant_(self.norm_out.linear.weight, 0)
            nn.init.constant_(self.norm_out.linear.bias, 0)
        elif self.config.timestep_conditioning == "adaln-single":
            nn.init.normal_(self.t_block[1].weight, std=self.config.base_config.initializer_range)
        
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        if self.config.timestep_conditioning is not None:
            nn.init.normal_(self.timestep_embedder[0].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.timestep_embedder[2].weight, std=self.config.base_config.initializer_range)

        if self.config.text_modulation_embeds_dim is not None:
            nn.init.normal_(self.condition_embedder[1].weight, std=self.config.base_config.initializer_range)
            nn.init.normal_(self.condition_embedder[3].weight, std=self.config.base_config.initializer_range)
        
        nn.init.normal_(self.context_embedder[1].weight, std=self.config.base_config.initializer_range)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = { "use_reentrant": True }
        gradient_checkpointing_func = functools.partial(torch.utils.checkpoint.checkpoint, **gradient_checkpointing_kwargs)
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

from transformers import Qwen3VLTextModel



def build_dit(hparams):
    """Build a standard DiT model."""
    base_config = AutoConfig.from_pretrained(hparams.model.base)
    load_model = get_llm(hparams.model.base, base_config)
    base_config = load_model.config
    base_config = _patch_base_config_defaults(base_config)
    del load_model

    assert hparams.model.pos_embed in ["ape", "1d-rope", "2d-rope", "m-rope"]
    assert hparams.model.timestep_conditioning in [None, "adaln-zero", "adaln-single", "addition"]

    if not hasattr(base_config, "head_dim"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads

    config = DiTConfig(
        attention=hparams.model.attention if hasattr(hparams.model, "attention") else "self",
        base_config=base_config,
        dit_hidden_size=hparams.model.dit_hidden_size if hasattr(hparams.model, "dit_hidden_size") else base_config.hidden_size,
        dit_num_hidden_layers=hparams.model.dit_num_hidden_layers,
        patch_size=hparams.model.patch_size,
        pos_embed=hparams.model.pos_embed,
        qk_norm=hparams.model.qk_norm if hasattr(hparams.model, "qk_norm") else False,
        text_hidden_size=base_config.hidden_size,
        text_hidden_states_index=hparams.model.text_hidden_states_index if hasattr(hparams.model, "text_hidden_states_index") else -1,
        text_modulation_embeds_dim=hparams.model.text_modulation_embeds_dim if hasattr(hparams.model, "text_modulation_embeds_dim") else None,
        timestep_conditioning=hparams.model.timestep_conditioning,
        use_norm_avg_fusion=hparams.model.use_norm_avg_fusion if hasattr(hparams.model, "use_norm_avg_fusion") else False,
    )
    
    transformer = DiT(config)
    transformer.requires_grad_(hparams.trainer.train_dit)
    
    return transformer


def build_adafusedit(hparams):
    """Build an AdaFuseDiT model with multi-layer text fusion."""
    base_config = AutoConfig.from_pretrained(hparams.model.base)
    load_model = get_llm(hparams.model.base, base_config)
    base_config = load_model.config
    base_config = _patch_base_config_defaults(base_config)
    del load_model

    assert hparams.model.pos_embed in ["ape", "1d-rope", "2d-rope", "m-rope"]
    assert hparams.model.timestep_conditioning in [None, "adaln-zero", "adaln-single", "addition"]

    if not hasattr(base_config, "head_dim"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads

    config = DiTConfig(
        attention=hparams.model.attention if hasattr(hparams.model, "attention") else "self",
        base_config=base_config,
        dit_hidden_size=hparams.model.dit_hidden_size if hasattr(hparams.model, "dit_hidden_size") else base_config.hidden_size,
        dit_num_hidden_layers=hparams.model.dit_num_hidden_layers,
        patch_size=hparams.model.patch_size,
        pos_embed=hparams.model.pos_embed,
        qk_norm=hparams.model.qk_norm if hasattr(hparams.model, "qk_norm") else False,
        text_hidden_size=base_config.hidden_size,
        text_hidden_states_index=hparams.model.text_hidden_states_index if hasattr(hparams.model, "text_hidden_states_index") else -1,
        text_modulation_embeds_dim=hparams.model.text_modulation_embeds_dim if hasattr(hparams.model, "text_modulation_embeds_dim") else None,
        timestep_conditioning=hparams.model.timestep_conditioning,
        # AdaFuseDiT specific settings
        text_hidden_states_num=hparams.model.text_hidden_states_num if hasattr(hparams.model, "text_hidden_states_num") else 1,
        use_timestep_adaptive_fusion=hparams.model.use_timestep_adaptive_fusion if hasattr(hparams.model, "use_timestep_adaptive_fusion") else False,
        use_layer_wise_fusion=hparams.model.use_layer_wise_fusion if hasattr(hparams.model, "use_layer_wise_fusion") else False,
        adaptive_fusion_time_embed_dim=hparams.model.adaptive_fusion_time_embed_dim if hasattr(hparams.model, "adaptive_fusion_time_embed_dim") else 128,
    )
    
    transformer = AdaFuseDiT(config)
    transformer.requires_grad_(hparams.trainer.train_dit)
    
    return transformer


def build_fusedit(hparams):
    base_config = AutoConfig.from_pretrained(hparams.model.base)
    load_model = get_llm(hparams.model.base, base_config)
    base_config = load_model.config
    base_config = _patch_base_config_defaults(base_config)

    assert hparams.model.attention in ["self", "cross"]
    assert hparams.model.pos_embed in ["ape", "1d-rope", "2d-rope", "m-rope"]
    assert hparams.model.timestep_conditioning in [None, "adaln-zero", "adaln-single", "addition"]

    if not hasattr(base_config, "head_dim"):
        base_config.head_dim = base_config.hidden_size // base_config.num_attention_heads
    #base_config.num_hidden_layers = min(
    #    base_config.num_hidden_layers,
    #    max(hparams.model.shared_attention_layers) + 1 if hparams.model.shared_attention_layers != "all" else base_config.num_hidden_layers,
    #)

    config = FuseDiTConfig(
        attention=hparams.model.attention,
        base_config=base_config,
        dit_hidden_size=hparams.model.dit_hidden_size if hasattr(hparams.model, "dit_hidden_size") else base_config.hidden_size,
        dit_num_hidden_layers=hparams.model.dit_num_hidden_layers,
        initial_layers=hparams.model.initial_layers if hasattr(hparams.model, "initial_layers") else 0,
        patch_size=hparams.model.patch_size,
        pos_embed=hparams.model.pos_embed,
        qk_norm=hparams.model.qk_norm,
        sandwich_norm=hparams.model.sandwich_norm,
        shared_attention_layers=hparams.model.shared_attention_layers,
        text_hidden_size=base_config.hidden_size,
        text_modulation_embeds_dim=hparams.model.text_modulation_embeds_dim if hasattr(hparams.model, "text_modulation_embeds_dim") else None,
        timestep_conditioning=hparams.model.timestep_conditioning,
    )
    transformer = FuseDiT(config)

    transformer.llm.load_state_dict(load_model.state_dict(), strict=False)
    del load_model

    transformer.dit.requires_grad_(hparams.trainer.train_dit)
    transformer.llm.requires_grad_(hparams.trainer.train_llm)

    return transformer



def build_model(hparams):
    if hparams.model.name == "DiT":
        return build_dit(hparams)
    elif hparams.model.name == "AdaFuseDiT":
        return build_adafusedit(hparams)
    elif hparams.model.name == "FuseDiT":
        return build_fusedit(hparams)
