"""Drama-inspired Mamba-2 dynamics for Memoir, using functional PyTorch state updates.

This is an independent functional implementation of the Mamba-2 equations in
state-spaces/mamba v2.2.4 (Apache-2.0). It is not a GRU or a Transformer, and does
not claim the performance of the fused CUDA implementation. Functional caches
are essential here: Memoir backpropagates through recurrent steps and forks
overshooting/imagination branches, whereas official inference step() mutates
its cache in place. Drama's two-layer, width-512, state-16 backbone is adapted
to continuous VLN latents and language conditioning, not its Atari actor or
replay-sampling algorithm. See DRAMA_MAMBA2.md for provenance and deviations.
"""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


WORLD_MODEL_PREFIXES = ('lang_init.', 'prior.', 'posterior.', 'to_vis_embed.',
                        'to_reward.', 'obs_transform.')
TEMPORAL_HYBRID_BACKEND = 'drama_mamba2_temporal_hybrid'
TEMPORAL_LOOP_BACKEND = 'drama_mamba2_temporal_loop'


class GatedRMSNorm(nn.Module):
    """Mamba-2's norm_before_gate=False, ngroups=1 configuration."""
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value, gate):
        x = value.float() * F.silu(gate.float())
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(value.dtype)


class FunctionalMamba2(nn.Module):
    """Selective SSM with head-shared A, input-dependent B/C/dt and causal conv.

    Cache shapes are (B, inner+2*state, conv_width) and
    (B, heads, head_dim, state). SSM accumulation uses FP32 under autocast.
    No cache is detached or updated in place, so all history remains learnable.
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2, headdim=64):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner, self.headdim = d_model * expand, headdim
        if self.d_inner % headdim or min(d_model, d_state, d_conv, headdim) < 1:
            raise ValueError('Invalid Mamba-2 dimensions')
        self.nheads = self.d_inner // headdim
        self.conv_dim = self.d_inner + 2 * d_state
        self.in_proj = nn.Linear(d_model, 2*self.d_inner + 2*d_state + self.nheads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, d_conv,
                              groups=self.conv_dim, padding=d_conv-1)
        # HF's no_init_weights suppresses nn.init while constructing from a
        # checkpoint. BERT's missing-key initializer does not handle Conv1d.
        # Initialize with Tensor.uniform_ so this new depthwise convolution is
        # valid in both ordinary and HF factories. Loaded checkpoints still
        # replace these values afterwards. This is PyTorch's default fan-in
        # uniform distribution for a depthwise convolution (fan_in=d_conv).
        with torch.no_grad():
            bound = 1 / math.sqrt(d_conv)
            self.conv1d.weight.uniform_(-bound, bound)
            self.conv1d.bias.uniform_(-bound, bound)
        dt = torch.exp(torch.empty(self.nheads).uniform_(math.log(.001), math.log(.1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = GatedRMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True

    def step(self, u, cache=None):
        if u.ndim != 2 or u.shape[-1] != self.d_model:
            raise ValueError('Mamba-2 step expects (batch, d_model)')
        batch = u.shape[0]
        projected = self.in_proj(u)
        z, xbc, dt_raw = projected.split((self.d_inner, self.conv_dim, self.nheads), -1)
        if cache is None:
            conv = xbc.new_zeros(batch, self.conv_dim, self.d_conv)
            state = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                                device=u.device, dtype=torch.float32)
        else:
            conv, state = cache
            if conv.shape != (batch, self.conv_dim, self.d_conv) or state.shape != (
                    batch, self.nheads, self.headdim, self.d_state):
                raise ValueError('Mamba-2 cache shape does not match current batch')
        conv = torch.cat((conv[..., 1:], xbc.unsqueeze(-1)), -1)
        xbc = (conv * self.conv1d.weight[:, 0, :]).sum(-1)
        xbc = F.silu(xbc + self.conv1d.bias)
        x, b, c = xbc.split((self.d_inner, self.d_state, self.d_state), -1)
        # Disable autocast for the recurrent state. Stable accumulation matters
        # for long tours and avoids state dtype depending on the caller context.
        context = torch.autocast(device_type=u.device.type, enabled=False) if u.device.type in ('cpu', 'cuda') else nullcontext()
        with context:
            dt = F.softplus(dt_raw.float() + self.dt_bias.float())
            decay = torch.exp(-dt * self.A_log.float().exp())
            xh = x.float().reshape(batch, self.nheads, self.headdim)
            state = state.float() * decay[..., None, None] + (
                dt[..., None, None] * xh[..., None] * b.float()[:, None, None, :])
            y = (state * c.float()[:, None, None, :]).sum(-1)
            y = y + self.D.float()[None, :, None] * xh
        y = self.norm(y.reshape(batch, self.d_inner).to(projected.dtype), z)
        return self.out_proj(y), (conv, state)

    def forward(self, inputs, cache=None):
        if inputs.ndim != 3 or inputs.shape[1] == 0:
            raise ValueError('Mamba-2 forward expects a nonempty (B,T,D) sequence')
        outputs = []
        for token in inputs.unbind(1):
            output, cache = self.step(token, cache)
            outputs.append(output)
        return torch.stack(outputs, 1), cache


class CausalTemporalAttention(nn.Module):
    """Full causal self-attention over trajectory positions, with T5 distance buckets."""

    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('Temporal attention width must be divisible by its head count')
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.relative_bias = nn.Embedding(32, n_heads)

    def _relative_buckets(self, distance):
        # Single-direction T5 buckets: exact lags 0..15, logarithmic to 128.
        exact = self.relative_bias.num_embeddings // 2
        long_bucket = torch.log(distance.clamp_min(exact).float() / exact) / math.log(128 / exact)
        buckets = torch.where(distance < exact, distance, exact + (long_bucket * exact).long())
        return buckets.clamp(0, self.relative_bias.num_embeddings - 1)

    def forward(self, inputs, cache=None):
        batch, steps, width = inputs.shape
        q, k, v = self.qkv(inputs).view(batch, steps, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        past = 0
        if cache is not None:
            past_k, past_v = cache
            if (past_k.ndim != 4 or past_v.shape != past_k.shape
                    or past_k.shape[:2] != (batch, self.n_heads) or past_k.shape[-1] != self.head_dim):
                raise ValueError('Temporal KV cache shape does not match current batch or heads')
            past = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        q_pos = torch.arange(past, past + steps, device=inputs.device)
        k_pos = torch.arange(k.size(2), device=inputs.device)
        distance = q_pos[:, None] - k_pos[None, :]
        bias = self.relative_bias(self._relative_buckets(distance)).permute(2, 0, 1)
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
        scores = (scores.float() + bias.float()).masked_fill(distance < 0, float('-inf'))
        attended = scores.softmax(-1).to(v.dtype) @ v
        output = self.out_proj(attended.transpose(1, 2).reshape(batch, steps, width))
        return output, (k, v)


    def refine(self, inputs, norm, dropout, cache=None, iterations=2):
        """Refine queries against fixed source K/V without advancing time again.

        The first pass is exactly the ordinary pre-norm attention residual.
        Subsequent passes update only the query/residual representation. K/V
        are derived from the pre-refinement source once and appended once per
        trajectory position, including in full-sequence and chunked execution.
        This is fixed-memory looped attention, with no added FFN or parameters.
        """
        if type(iterations) is not int or iterations < 1:
            raise ValueError('Loop iterations must be a positive integer')
        delta, memory = self(norm(inputs), cache)
        x = inputs + dropout(delta)
        if iterations == 1:
            return x, memory
        k, v = memory
        batch, steps, width = inputs.shape
        past = 0 if cache is None else cache[0].size(2)
        q_pos = torch.arange(past, past + steps, device=inputs.device)
        k_pos = torch.arange(k.size(2), device=inputs.device)
        distance = q_pos[:, None] - k_pos[None, :]
        bias = self.relative_bias(self._relative_buckets(distance)).permute(2, 0, 1)
        for _ in range(1, iterations):
            # Slice the shared Q weights; never re-project or append source K/V.
            q = F.linear(norm(x), self.qkv.weight[:width])
            q = q.view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5)
            scores = (scores.float() + bias.float()).masked_fill(distance < 0, float('-inf'))
            attended = scores.softmax(-1).to(v.dtype) @ v
            delta = self.out_proj(attended.transpose(1, 2).reshape(batch, steps, width))
            x = x + dropout(delta)
        return x, memory


class DramaMamba2Prior(nn.Module):
    """Language-conditioned causal Mamba-2 replacing only Memoir's TSSM prior.

    The four-tensor layer tuple preserves the author's overshooting batching
    interface. Its first two slots are convolution/SSM state in Mamba layers
    and trajectory keys/values in the optional fourth attention layer. The last
    two slots always hold projections of the fixed instruction.
    """
    def __init__(self, stoch_size=96, stoch_discrete=False, hidden_size=768,
                 deter_size=672, d_model=512, n_layers=2, d_state=16,
                 headdim=64, dropout=0.1, temporal_attention=False, loop_iterations=None):
        super().__init__()
        if temporal_attention and (n_layers, d_model) != (4, 768):
            raise ValueError('The temporal hybrid requires four layers of width 768')
        if loop_iterations is not None:
            if not temporal_attention or type(loop_iterations) is not int or loop_iterations < 1:
                raise ValueError('Loop refinement requires temporal attention and a positive integer depth')
            # Persist depth so raw state_dict exports cannot masquerade as an
            # ordinary hybrid, or silently load with a different loop count.
            self.register_buffer('_loop_iterations', torch.tensor(loop_iterations, dtype=torch.int64))
        self.loop_iterations = loop_iterations
        self.hidden_size, self.deter_size = hidden_size, deter_size
        self.d_model, self.n_layers = d_model, n_layers
        latent = stoch_size * (stoch_discrete or 1) + deter_size
        self.input_proj = nn.Linear(latent, d_model)
        self.language_q = nn.Linear(d_model, d_model, bias=False)
        self.language_k = nn.Linear(hidden_size, d_model, bias=False)
        self.language_v = nn.Linear(hidden_size, d_model, bias=False)
        self.language_o = nn.Linear(d_model, d_model, bias=False)
        self.language_norm = nn.LayerNorm(d_model)
        self.pre_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        n_mamba = n_layers - int(temporal_attention)
        self.layers = nn.ModuleList([FunctionalMamba2(d_model, d_state, headdim=headdim)
                                     for _ in range(n_mamba)])
        if temporal_attention:
            self.layers.append(CausalTemporalAttention(d_model, n_heads=12))
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.to_deter = nn.Linear(d_model, deter_size)
        self.dense = nn.Sequential(nn.Linear(deter_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self._norm = nn.LayerNorm(hidden_size)
        self.mu_std = nn.Linear(hidden_size, stoch_size * (stoch_discrete or 2))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        marker = state_dict.get(prefix + '_loop_iterations')
        if marker is not None and (self.loop_iterations is None or not isinstance(marker, torch.Tensor)
                or marker.dtype != torch.int64 or marker.ndim != 0
                or marker.item() != self.loop_iterations):
            error_msgs.append('Checkpoint loop depth differs from the configured temporal backbone')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, prev_states, encoder_embeds, encoder_attention_mask, past_key_values=None):
        if prev_states.ndim != 3 or prev_states.shape[1] < 1:
            raise ValueError('Dynamics requires nonempty (B,T,768) states')
        if encoder_embeds.ndim != 3 or encoder_embeds.shape[0] != prev_states.shape[0]:
            raise ValueError('Instruction/state batch mismatch')
        mask = encoder_attention_mask.bool()
        if mask.shape != encoder_embeds.shape[:2] or not bool(mask.any(-1).all()):
            raise ValueError('Instruction mask must have at least one valid token per row')
        cache = past_key_values
        if cache is not None and len(cache) != self.n_layers:
            raise ValueError('Mamba-2 layer cache count mismatch')
        if cache is None:
            keys, values = self.language_k(encoder_embeds), self.language_v(encoder_embeds)
        else:
            keys, values = cache[0][2], cache[0][3]
            if keys.shape != (prev_states.shape[0], encoder_embeds.shape[1], self.d_model):
                raise ValueError('Cached language has incompatible batch/length')
        x = self.input_proj(prev_states)
        scores = torch.matmul(self.language_q(x), keys.transpose(-1, -2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min)
        context = torch.matmul(scores.float().softmax(-1).to(values.dtype), values)
        x = self.language_norm(x + self.language_o(context))
        new_cache = []
        for i, (norm, layer) in enumerate(zip(self.pre_norms, self.layers)):
            layer_cache = None if cache is None else cache[i][:2]
            if self.loop_iterations is not None and i == self.n_layers - 1:
                x, state = layer.refine(x, norm, self.dropout, layer_cache, self.loop_iterations)
            else:
                y, state = layer(norm(x), layer_cache)
                x = x + self.dropout(y)
            new_cache.append((state[0], state[1], keys, values))
        deter = self.to_deter(self.final_norm(x[:, -1]))
        statistics = self.mu_std(F.gelu(self._norm(self.dense(deter))))
        return statistics, deter, tuple(new_cache)


def freeze_encoders(model):
    """Freeze text, panorama, and their shared token-type embeddings completely."""
    for name in ('embeddings', 'lang_encoder', 'img_embeddings'):
        module = getattr(model, name)
        module.requires_grad_(False)
        module.eval()
    model.config.freeze_encoders = True


def freeze_for_world_model_pretraining(model):
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(WORLD_MODEL_PREFIXES):
            parameter.requires_grad_(True)
    freeze_encoders(model)
    return [name for name, p in model.named_parameters() if p.requires_grad]
