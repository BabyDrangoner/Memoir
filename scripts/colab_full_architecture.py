"""World-model identities for the full-data training and checkpoint handoff."""

MAMBA2 = "drama_mamba2"
TEMPORAL_HYBRID = "drama_mamba2_temporal_hybrid"
BACKENDS = (MAMBA2, TEMPORAL_HYBRID)
ARCH = {"d_model": 768, "layers": 4, "d_state": 16, "headdim": 64}


def architecture(config, backend):
    dims = {key: config[key] for key in ARCH}
    if backend == MAMBA2:
        # Earlier pure Mamba-2 handoffs contain just these four fields.
        return dims
    if backend != TEMPORAL_HYBRID or dims != ARCH:
        raise ValueError(f"Unsupported temporal architecture: {backend}, {dims}")
    return {
        **dims,
        "world_model_backend": TEMPORAL_HYBRID,
        "temporal_layers": ["mamba2", "mamba2", "mamba2", "causal_full_attention"],
        "temporal_attention_heads": 12,
    }
