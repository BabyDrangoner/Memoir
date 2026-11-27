"""World-model identities for the full-data training and checkpoint handoff."""

MAMBA2 = "drama_mamba2"
TEMPORAL_HYBRID = "drama_mamba2_temporal_hybrid"
TEMPORAL_LOOP = "drama_mamba2_temporal_loop"
BACKENDS = (MAMBA2, TEMPORAL_HYBRID, TEMPORAL_LOOP)
ARCH = {"d_model": 768, "layers": 4, "d_state": 16, "headdim": 64}


def architecture(config, backend):
    dims = {key: config[key] for key in ARCH}
    if backend != TEMPORAL_LOOP and "loop_iterations" in config:
        raise ValueError("loop_iterations belongs only to the temporal loop backend")
    if backend == MAMBA2:
        # Earlier pure Mamba-2 handoffs contain just these four fields.
        return dims
    if backend not in (TEMPORAL_HYBRID, TEMPORAL_LOOP) or dims != ARCH:
        raise ValueError(f"Unsupported temporal architecture: {backend}, {dims}")
    result = {
        **dims,
        "world_model_backend": backend,
        "temporal_layers": ["mamba2", "mamba2", "mamba2", "causal_full_attention"],
        "temporal_attention_heads": 12,
    }
    if backend == TEMPORAL_LOOP:
        loops = config.get("loop_iterations")
        if type(loops) is not int or loops < 1:
            raise ValueError("The loop backend requires a positive integer loop_iterations")
        result.update(loop_iterations=loops,
                      temporal_layers=["mamba2"] * 3 + ["loop_fixed_memory_attention"],
                      loop_memory="fixed_source_kv_per_time_step",
                      loop_updates="shared_query_latent_refinement",
                      loop_ffn=False)
    return result
