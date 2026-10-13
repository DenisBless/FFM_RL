import jax
import jax.numpy as jnp
from flax import nnx
from typing import Literal, Optional, Tuple


class FeatureProjector(nnx.Module):
    """
    Helper module to process raw inputs (time or actions) into embeddings.
    Supports:
    - Linear vs MLP mode.
    - Fourier features with configurable frequency ranges.
    - Mixing linear and fourier features as MLP inputs.
    """

    def __init__(
        self,
        input_dim: int,
        mode: Literal["linear", "mlp"] = "mlp",
        mlp_input_type: Literal["linear", "fourier", "both"] = "both",
        num_fourier_feats: int = 32,
        fourier_range: Tuple[float, float] = (0.1, 100.0),
        num_hid: int = 64,
        num_out: int = 16,
        *,
        rngs: nnx.Rngs,
    ):
        self.mode = mode
        self.mlp_input_type = mlp_input_type
        self.input_dim = input_dim

        if mode == "linear":
            self.output_dim = input_dim
            return

        self.num_fourier_feats = num_fourier_feats
        freqs = jnp.geomspace(fourier_range[0], fourier_range[1], num_fourier_feats)
        # singleton dim for broadcasting with input_dim
        self.coeff = nnx.Variable(freqs[None, :])
        # Leave phase trainable per input_dim
        self.phase = nnx.Param(jnp.zeros((input_dim, num_fourier_feats)))

        # sin and cos per frequency per input dim
        fourier_dim = input_dim * num_fourier_feats * 2

        if mlp_input_type == "linear":
            mlp_in_dim = input_dim
        elif mlp_input_type == "fourier":
            mlp_in_dim = fourier_dim
        else:  # "both"
            mlp_in_dim = fourier_dim + input_dim

        self.mlp = nnx.Sequential(
            nnx.Linear(mlp_in_dim, num_hid, rngs=rngs),
            nnx.gelu,
            nnx.Linear(num_hid, num_out, rngs=rngs),
        )
        self.output_dim = num_out

    def _get_fourier(self, x):
        # x shape: (input_dim,)
        # Reshape for broadcasting with coeff (input_dim, freqs)
        x_expanded = x[:, None]
        angle = (x_expanded * self.coeff.value) + self.phase.value
        # (input_dim, num_fourier_feats * 2)
        feats = jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)
        # Flatten to (input_dim * num_fourier_feats * 2)
        return feats.reshape(-1)

    def __call__(self, x):
        # x may be scalar instead of 1D
        x = jnp.atleast_1d(x)

        if self.mode == "linear":
            return x

        # Prepare MLP input
        if self.mlp_input_type == "linear":
            mlp_in = x
        elif self.mlp_input_type == "fourier":
            mlp_in = self._get_fourier(x)
        else:  # both
            mlp_in = jnp.concatenate([x, self._get_fourier(x)], axis=-1)

        return self.mlp(mlp_in)


class ControlNetwork(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        num_layers: int = 2,
        num_hid: int = 64,
        # Time configuration
        time_mode: Literal["linear", "mlp"] = "mlp",
        time_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_time_fourier: int = 16,
        time_fourier_range_min: float = 0.1,
        time_fourier_range_max: float = 100.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        # Action configuration
        action_mode: Literal["linear", "mlp"] = "linear",
        action_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_action_fourier: int = 16,
        action_fourier_range_min: float = 0.1,
        action_fourier_range_max: float = 100.0,
        num_action_hid: int = 32,
        num_action_out: int = 16,
        outer_clip: float = 1e4,
        inner_clip: float = 1e2,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        layer_norm: bool = False,
        layer_norm_type: str = "LayerNorm",
        *,
        rngs: nnx.Rngs,
    ):
        self.outer_clip = outer_clip

        time_fourier_range = (time_fourier_range_min, time_fourier_range_max)
        # Time encoder network
        self.time_projector = FeatureProjector(
            input_dim=1,
            mode=time_mode,
            mlp_input_type=time_mlp_input,
            num_fourier_feats=num_time_fourier,
            fourier_range=time_fourier_range,
            num_hid=num_time_hid,
            num_out=num_time_out,
            rngs=rngs,
        )

        action_fourier_range = (action_fourier_range_min, action_fourier_range_max)
        # Action encoder network
        self.action_projector = FeatureProjector(
            input_dim=action_dim,
            mode=action_mode,
            mlp_input_type=action_mlp_input,
            num_fourier_feats=num_action_fourier,
            fourier_range=action_fourier_range,
            num_hid=num_action_hid,
            num_out=num_action_out,
            rngs=rngs,
        )

        # State-time network
        # Dynamically calculate input size
        combined_input_dim = self.action_projector.output_dim + observation_dim + self.time_projector.output_dim

        layers = []
        curr_dim = combined_input_dim

        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_dim = action_dim if is_last else num_hid

            layer = nnx.Linear(curr_dim, out_dim, rngs=rngs)

            if is_last:
                # custom init but no norm or activation afterwards
                layer.kernel.value *= weight_init
                layer.bias.value = jnp.full_like(layer.bias.value, bias_init)
                layers.append(layer)
            else:
                layers.append(layer)
                if layer_norm:
                    layers.append(getattr(nnx, layer_norm_type)(out_dim, rngs=rngs))
                layers.append(nnx.gelu)

            curr_dim = out_dim

        self.state_time_net = nnx.Sequential(*layers)

    def __call__(self, actions, observations, time):
        """
        actions: (action_dim)
        observations: (obs_dim)
        time: () or (1,) scalar in [0, 1]
        """
        t_scaled = (time * 2.0) - 1.0

        t_emb = self.time_projector(t_scaled)
        a_emb = self.action_projector(actions)

        x = jnp.concatenate([a_emb, observations, t_emb], axis=-1)
        out = self.state_time_net(x)

        return jnp.clip(out, -self.outer_clip, self.outer_clip)

class ResidualBlock(nnx.Module):
    """
    A modern residual block supporting AdaLN conditioning, Gated GELU, 
    and SimBAv2-style magnitude shift preservation.
    """
    def __init__(
        self,
        stream_dim: int,
        context_dim: int | None = None,
        cond_mode: Literal["concat", "adaln"] = "adaln",
        activation: Literal["gelu", "gated_gelu"] = "gated_gelu",
        layer_norm_type: str = "RMSNorm",
        simba_shift: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.cond_mode = cond_mode
        self.activation = activation
        self.simba_shift = simba_shift

        # Pre-Normalization layer
        self.norm = getattr(nnx, layer_norm_type)(stream_dim, rngs=rngs)

        # AdaLN Modulation Projections
        if cond_mode == "adaln":
            assert context_dim is not None, "context_dim must be provided for adaln mode."
            self.adaln_proj = nnx.Linear(
                context_dim, 
                2 * stream_dim, 
                kernel_init=jax.nn.initializers.zeros, # Initialize to 0 for stable start
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs
            )

        # Simbav2 prepends/appends a constant 1 to the features to retain magnitude shift
        mlp_in_dim = stream_dim + 1 if simba_shift else stream_dim

        # MLP Core
        if activation == "gelu":
            hidden_dim = int(4 * stream_dim) # Standard 4d expansion
            self.up_proj = nnx.Linear(mlp_in_dim, hidden_dim, rngs=rngs)
        elif activation == "gated_gelu":
            hidden_dim = int((8 / 3) * stream_dim) # Parameter-matched expansion
            self.gate_proj = nnx.Linear(mlp_in_dim, hidden_dim, rngs=rngs)
            self.up_proj = nnx.Linear(mlp_in_dim, hidden_dim, rngs=rngs)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Final projection: initialized to zero so the block is exactly the identity function at init
        self.down_proj = nnx.Linear(
            hidden_dim, 
            stream_dim, 
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs
        )

    def __call__(self, x, context=None):
        # 1. Pre-Norm
        h = self.norm(x)

        # 2. Conditioning
        if self.cond_mode == "adaln":
            scale_shift = self.adaln_proj(context)
            scale, shift = jnp.split(scale_shift, 2, axis=-1)
            h = h * (1.0 + scale) + shift

        # 3. SimBAv2 magnitude preserving shift (appends a 1 to the feature vector)
        if self.simba_shift:
            # Handles unbatched (dim,) and batched (B, dim) automatically
            ones = jnp.ones((*h.shape[:-1], 1), dtype=h.dtype)
            h = jnp.concatenate([h, ones], axis=-1)

        # 4. Activation Expansion
        if self.activation == "gelu":
            h = self.up_proj(h)
            h = nnx.gelu(h)
        elif self.activation == "gated_gelu":
            gate = nnx.gelu(self.gate_proj(h))
            up = self.up_proj(h)
            h = gate * up

        # 5. Down-projection & Residual connection
        h = self.down_proj(h)
        return x + h


class ResidualControlNetwork(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        # Residual Architecture config
        stream_dim: int = 128,
        num_blocks: int = 4,
        cond_mode: Literal["concat", "adaln"] = "adaln",
        context_dim: int = 128,
        activation: Literal["gelu", "gated_gelu"] = "gated_gelu",
        layer_norm_type: str = "RMSNorm",  # Changed default to RMSNorm for modern standards
        simba_shift: bool = False,
        # Time configuration
        time_mode: Literal["linear", "mlp"] = "mlp",
        time_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_time_fourier: int = 16,
        time_fourier_range_min: float = 0.1,
        time_fourier_range_max: float = 100.0,
        num_time_hid: int = 32,
        num_time_out: int = 16,
        # Action configuration
        action_mode: Literal["linear", "mlp"] = "linear",
        action_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_action_fourier: int = 16,
        action_fourier_range_min: float = 0.1,
        action_fourier_range_max: float = 100.0,
        num_action_hid: int = 32,
        num_action_out: int = 16,
        # Output config
        outer_clip: float = 1e4,
        weight_init: float = 1e-8,
        bias_init: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.outer_clip = outer_clip
        self.cond_mode = cond_mode

        # Time encoder network
        self.time_projector = FeatureProjector(
            input_dim=1,
            mode=time_mode,
            mlp_input_type=time_mlp_input,
            num_fourier_feats=num_time_fourier,
            fourier_range=(time_fourier_range_min, time_fourier_range_max),
            num_hid=num_time_hid,
            num_out=num_time_out,
            rngs=rngs,
        )

        # Action encoder network
        self.action_projector = FeatureProjector(
            input_dim=action_dim,
            mode=action_mode,
            mlp_input_type=action_mlp_input,
            num_fourier_feats=num_action_fourier,
            fourier_range=(action_fourier_range_min, action_fourier_range_max),
            num_hid=num_action_hid,
            num_out=num_action_out,
            rngs=rngs,
        )

        # Determine architecture topology based on Conditioning Mode
        if cond_mode == "concat":
            combined_input_dim = self.action_projector.output_dim + observation_dim + self.time_projector.output_dim
            self.initial_proj = nnx.Linear(combined_input_dim, stream_dim, rngs=rngs)
            
        elif cond_mode == "adaln":
            # For AdaLN, we process the action/state into the main residual stream.
            self.initial_proj = nnx.Linear(self.action_projector.output_dim, stream_dim, rngs=rngs)
            
            # Shared MLP processing observations and time to build a contextual representation
            context_input_dim = observation_dim + self.time_projector.output_dim
            self.context_mlp = nnx.Sequential(
                nnx.Linear(context_input_dim, context_dim, rngs=rngs),
                nnx.gelu,
                nnx.Linear(context_dim, context_dim, rngs=rngs)
            )
        else:
            raise ValueError(f"Unknown condition mode: {cond_mode}")

        # Construct Residual Blocks
        self.blocks =[
            ResidualBlock(
                stream_dim=stream_dim,
                context_dim=context_dim if cond_mode == "adaln" else None,
                cond_mode=cond_mode,
                activation=activation,
                layer_norm_type=layer_norm_type,
                simba_shift=simba_shift,
                rngs=rngs
            )
            for _ in range(num_blocks)
        ]

        # Final projection sequence
        self.final_norm = getattr(nnx, layer_norm_type)(stream_dim, rngs=rngs)
        self.final_proj = nnx.Linear(stream_dim, action_dim, rngs=rngs)
        
        # Apply custom initialization to the last layer so the initial network predictions are close to zero
        self.final_proj.kernel.value *= weight_init
        self.final_proj.bias.value = jnp.full_like(self.final_proj.bias.value, bias_init)

    def __call__(self, actions, observations, time):
        """
        actions: (action_dim)
        observations: (obs_dim)
        time: () or (1,) scalar in[0, 1]
        """
        t_scaled = (time * 2.0) - 1.0

        t_emb = self.time_projector(t_scaled)
        a_emb = self.action_projector(actions)

        # 1. Start the feature stream
        if self.cond_mode == "concat":
            x = jnp.concatenate([a_emb, observations, t_emb], axis=-1)
            h = self.initial_proj(x)
            context = None
        else:  # adaln
            h = self.initial_proj(a_emb)
            c_in = jnp.concatenate([observations, t_emb], axis=-1)
            context = self.context_mlp(c_in)

        # 2. Pass through residual blocks
        for block in self.blocks:
            h = block(h, context)

        # 3. Final norm and projection
        h = self.final_norm(h)
        out = self.final_proj(h)

        return jnp.clip(out, -self.outer_clip, self.outer_clip)


# =============================================================================
# DiT-style velocity network
# =============================================================================

class _CrossAttention(nnx.Module):
    """Scaled dot-product multi-head attention supporting cross-attention.

    Operates on single (unbatched) sequences; the training loop vmaps over
    the batch.  Shapes: x (S, D_q), context (C, D_kv), output (S, D_q).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        *,
        rngs: nnx.Rngs,
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        kv_dim = kv_dim or embed_dim

        self.q_proj   = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.k_proj   = nnx.Linear(kv_dim,    embed_dim, use_bias=False, rngs=rngs)
        self.v_proj   = nnx.Linear(kv_dim,    embed_dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim,  embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        if context is None:
            context = x
        H, D = self.num_heads, self.head_dim
        S, C = x.shape[0], context.shape[0]

        q = self.q_proj(x).reshape(S, H, D)       # (S, H, D)
        k = self.k_proj(context).reshape(C, H, D)  # (C, H, D)
        v = self.v_proj(context).reshape(C, H, D)  # (C, H, D)

        # (H, S, C) attention scores
        attn = jnp.einsum("shd,chd->hsc", q, k) * (D ** -0.5)
        attn = jax.nn.softmax(attn, axis=-1)

        # (S, H*D) output
        out = jnp.einsum("hsc,chd->shd", attn, v).reshape(S, H * D)
        return self.out_proj(out)


class _DiTBlock(nnx.Module):
    """One DiT-style transformer block for the velocity field.

    Processing order per token sequence x (A, D):
      1.  [optional] Self-attention (action ↔ action)  – with pre-RMSNorm
      2.  Cross-attention (action → obs context)        – with pre-RMSNorm
      3.  AdaLN-modulated SwiGLU FFN                    – with pre-RMSNorm

    Time enters only via AdaLN; observation context enters via cross-attention.
    This separates the two conditioning signals instead of entangling them.
    """

    def __init__(
        self,
        stream_dim: int,
        context_dim: int,
        time_emb_dim: int,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        use_self_attn: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.use_self_attn = use_self_attn

        # Self-attention
        if use_self_attn:
            self.norm_sa  = nnx.RMSNorm(stream_dim, rngs=rngs)
            self.self_attn = _CrossAttention(stream_dim, num_heads, rngs=rngs)

        # Cross-attention (action tokens → obs context)
        self.norm_ca    = nnx.RMSNorm(stream_dim, rngs=rngs)
        self.cross_attn = _CrossAttention(stream_dim, num_heads, kv_dim=context_dim, rngs=rngs)

        # AdaLN: time_emb → scale, shift, gate (3×stream_dim)
        # Zero-init so the block is identity at initialisation.
        self.norm_ffn     = nnx.RMSNorm(stream_dim, rngs=rngs)
        self.ada_modulate = nnx.Linear(
            time_emb_dim, 3 * stream_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

        # SwiGLU FFN: hidden_dim chosen so parameter count matches a 4× standard FFN.
        # SwiGLU uses two up-projections, so hidden = (4 * D * 2/3), rounded to 64.
        hidden_dim = max(64, int(ffn_mult * stream_dim * 2 / 3) // 64 * 64)
        self.ffn_gate = nnx.Linear(stream_dim, hidden_dim, rngs=rngs)
        self.ffn_up   = nnx.Linear(stream_dim, hidden_dim, rngs=rngs)
        self.ffn_down = nnx.Linear(
            hidden_dim, stream_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,         # (A, stream_dim)
        context: jax.Array,   # (K, context_dim)
        time_emb: jax.Array,  # (time_emb_dim,)
    ) -> jax.Array:

        # 1. Self-attention
        if self.use_self_attn:
            x = x + self.self_attn(self.norm_sa(x))

        # 2. Cross-attention to obs context
        x = x + self.cross_attn(self.norm_ca(x), context)

        # 3. AdaLN + SwiGLU FFN
        scale, shift, gate = jnp.split(self.ada_modulate(time_emb), 3, axis=-1)
        h = self.norm_ffn(x) * (1.0 + scale) + shift
        h = self.ffn_down(jax.nn.silu(self.ffn_gate(h)) * self.ffn_up(h))
        x = x + gate * h

        return x


class DiTControlNetwork(nnx.Module):
    """Diffusion Transformer (DiT) velocity network for SI policies.

    Each action dimension is an independent **token** in a residual stream.
    Tokens cross-attend to the observation and are time-modulated via AdaLN,
    enabling fine-grained per-dimension conditioning.

    Key architectural advances over ControlNetwork / ResidualControlNetwork
    -----------------------------------------------------------------------
    * Cross-attention (action ↔ obs): each action dimension can attend to
      *different* parts of the observation, strictly more expressive than the
      global AdaLN conditioning used in ResidualControlNetwork.
    * Self-attention (action ↔ action): captures inter-joint dependencies
      (e.g., knee depends on hip) without hard-coding the coupling structure.
    * SwiGLU FFN: empirically outperforms GELU and gated-GELU in transformers.
    * Separate conditioning pathways: time → AdaLN, obs → cross-attention.
      This prevents obs noise from interfering with temporal dynamics.
    * Learned per-dimension positional embeddings (action_dim × stream_dim).

    Interface
    ---------
    Identical to ControlNetwork / ResidualControlNetwork:
      __call__(actions, observations, time) → velocity   (all single-sample)
    Set `score_model.residual: false` and `residual_type: "dit"` in the config
    to use this architecture, or instantiate directly.
    """

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        # Stream / block dimensions
        stream_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        ffn_mult: float = 4.0,
        use_self_attn: bool = True,
        # Observation context
        context_dim: int = 128,
        num_context_tokens: int = 1,
        # Time embedding (Fourier features + MLP)
        time_mode: Literal["linear", "mlp"] = "mlp",
        time_mlp_input: Literal["linear", "fourier", "both"] = "both",
        num_time_fourier: int = 16,
        time_fourier_range_min: float = 0.1,
        time_fourier_range_max: float = 100.0,
        num_time_hid: int = 64,
        num_time_out: int = 64,
        # Output
        outer_clip: float = 1e4,
        weight_init: float = 1e-8,
        *,
        rngs: nnx.Rngs,
    ):
        assert stream_dim % num_heads == 0, "stream_dim must be divisible by num_heads"
        self.action_dim = action_dim
        self.stream_dim = stream_dim
        self.outer_clip = outer_clip
        self.num_context_tokens = num_context_tokens

        # ------------------------------------------------------------------
        # Time encoder: scalar t → time_emb (num_time_out,)
        # ------------------------------------------------------------------
        self.time_projector = FeatureProjector(
            input_dim=1,
            mode=time_mode,
            mlp_input_type=time_mlp_input,
            num_fourier_feats=num_time_fourier,
            fourier_range=(time_fourier_range_min, time_fourier_range_max),
            num_hid=num_time_hid,
            num_out=num_time_out,
            rngs=rngs,
        )
        time_emb_dim = num_time_out if time_mode == "mlp" else 1

        # ------------------------------------------------------------------
        # Observation encoder: obs_dim → (num_context_tokens, context_dim)
        # ------------------------------------------------------------------
        self.obs_proj = nnx.Linear(observation_dim, num_context_tokens * context_dim, rngs=rngs)
        self.obs_norm = nnx.RMSNorm(context_dim, rngs=rngs)

        # ------------------------------------------------------------------
        # Action tokeniser: each dimension is independently projected and
        # combined with a learned positional embedding.
        # ------------------------------------------------------------------
        self.action_in = nnx.Linear(1, stream_dim, rngs=rngs)
        self.pos_emb   = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (action_dim, stream_dim))
        )

        # ------------------------------------------------------------------
        # DiT blocks
        # ------------------------------------------------------------------
        self.blocks = [
            _DiTBlock(
                stream_dim=stream_dim,
                context_dim=context_dim,
                time_emb_dim=time_emb_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                use_self_attn=use_self_attn,
                rngs=rngs,
            )
            for _ in range(num_blocks)
        ]

        # ------------------------------------------------------------------
        # Output head: (A, stream_dim) → (A,)
        # ------------------------------------------------------------------
        self.final_norm = nnx.RMSNorm(stream_dim, rngs=rngs)
        self.final_proj = nnx.Linear(stream_dim, 1, rngs=rngs)
        self.final_proj.kernel.value *= weight_init
        self.final_proj.bias.value    = jnp.zeros_like(self.final_proj.bias.value)

    def __call__(
        self,
        actions: jax.Array,       # (action_dim,)
        observations: jax.Array,  # (obs_dim,)
        time: jax.Array,          # () or (1,)
    ) -> jax.Array:

        # --- Time embedding -------------------------------------------
        t_scaled = jnp.atleast_1d(time) * 2.0 - 1.0
        time_emb = self.time_projector(t_scaled)              # (time_emb_dim,)

        # --- Observation context tokens --------------------------------
        obs_ctx = self.obs_proj(observations)                  # (K * context_dim,)
        obs_ctx = obs_ctx.reshape(self.num_context_tokens, -1) # (K, context_dim)
        obs_ctx = self.obs_norm(obs_ctx)                       # normalise keys/values

        # --- Action tokens  -------------------------------------------
        # Each dimension is its own token: (A, 1) → (A, stream_dim)
        x = self.action_in(actions[:, None])     # (A, stream_dim)
        x = x + self.pos_emb.value               # add learned positional embedding

        # --- Transformer blocks ---------------------------------------
        for block in self.blocks:
            x = block(x, obs_ctx, time_emb)

        # --- Output ---------------------------------------------------
        x   = self.final_norm(x)
        out = self.final_proj(x).squeeze(-1)     # (A,)
        return jnp.clip(out, -self.outer_clip, self.outer_clip)