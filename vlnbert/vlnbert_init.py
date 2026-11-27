# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import (BertConfig, BertTokenizer)
from vlnbert.vilmodel import GlocalTextPathNavCMT
from vlnbert.drama_mamba2 import TEMPORAL_HYBRID_BACKEND, TEMPORAL_LOOP_BACKEND, WORLD_MODEL_PREFIXES
from transformers import PretrainedConfig
import copy
import torch
import os


def get_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer


def select_drama_weights(weights, backend, initialize_world, loop_iterations=None):
    normalized = {k.removeprefix('bert.'): v for k, v in weights.items()}
    keys = set(normalized)
    is_loop = backend == TEMPORAL_LOOP_BACKEND
    if is_loop and (type(loop_iterations) is not int or loop_iterations < 1):
        raise ValueError('A loop checkpoint requires an explicit positive integer loop depth')
    marker = normalized.get('prior._loop_iterations')
    if 'prior._loop_iterations' in keys:
        if (not is_loop or not isinstance(marker, torch.Tensor) or marker.ndim != 0
                or marker.dtype != torch.int64 or marker.item() != loop_iterations):
            raise ValueError('Checkpoint loop identity differs from the requested backbone/depth')
    elif is_loop and any(k.startswith('prior.layers.0.') for k in keys):
        raise ValueError('Unmarked hybrid checkpoint is not a trained loop checkpoint')
    if any(k.startswith('prior.layers.0.') for k in keys):
        has_temporal = any(k.startswith('prior.layers.') and k.endswith(('.qkv.weight', '.relative_bias.weight'))
                           for k in keys)
        if has_temporal != (backend in (TEMPORAL_HYBRID_BACKEND, TEMPORAL_LOOP_BACKEND)):
            raise ValueError('Checkpoint and requested temporal backbone differ')
        if has_temporal:
            temporal_keys = {k for k in keys if k.startswith('prior.layers.3.')}
            expected = {'prior.layers.3.' + name for name in ('qkv.weight', 'out_proj.weight', 'relative_bias.weight')}
            layers = {k.split('.')[2] for k in keys if k.startswith('prior.layers.')}
            if layers != {'0', '1', '2', '3'} or temporal_keys != expected:
                raise ValueError('Incomplete or incompatible temporal hybrid checkpoint')
        return weights
    if not initialize_world:
        raise ValueError('A pretrained Drama checkpoint is required. Enable initialize_drama_world only for fresh pretraining.')
    return {k: v for k, v in weights.items() if not k.removeprefix('bert.').startswith(WORLD_MODEL_PREFIXES)}


def get_vlnbert_models(args, config=None):
    model_name_or_path = args.bert_ckpt_file
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        for k, v in ckpt_weights.items():
            if 'global_encoder' in k:
                new_k = k.replace('global_encoder', 'exp_encoder')
                new_ckpt_weights[new_k] = copy.deepcopy(v)
            elif 'global_sap_head' in k:
                new_k = k.replace('global_sap_head', 'exp_sap_head')
                new_ckpt_weights[new_k] = copy.deepcopy(v)
            new_ckpt_weights[k] = v

    if args.tokenizer == 'xlm':
        cfg_name = os.path.join('lxmert-base-uncased')
    else:
        cfg_name = os.path.join('bert-base-uncased')
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    
    vis_config.world_model_type = 'tssm'
    vis_config.world_model_backend = getattr(args, 'world_model_backend', 'tssm')
    vis_config.freeze_encoders = getattr(args, 'freeze_encoders', False)
    if vis_config.world_model_backend == TEMPORAL_LOOP_BACKEND:
        depth = getattr(args, 'drama_loop_iterations', 2)
        if type(depth) is not int or depth < 1:
            raise ValueError('drama_loop_iterations must be a positive integer')
        vis_config.drama_loop_iterations = depth
    elif getattr(args, 'drama_loop_iterations', None) is not None:
        raise ValueError('Loop iterations are only valid for the temporal loop backend')
    for option, default in [('drama_d_model', 512), ('drama_layers', 2),
                            ('drama_d_state', 16), ('drama_headdim', 64), ('drama_dropout', 0.1)]:
        setattr(vis_config, option, getattr(args, option, default))
    if vis_config.world_model_backend in ('drama_mamba2', TEMPORAL_HYBRID_BACKEND, TEMPORAL_LOOP_BACKEND):
        new_ckpt_weights = select_drama_weights(new_ckpt_weights, vis_config.world_model_backend,
                                                getattr(args, 'initialize_drama_world', False),
                                                getattr(vis_config, 'drama_loop_iterations', None))

    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2

    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels
    vis_config.glocal_fuse = args.fusion == 'dynamic'

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False

    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path=None,
        config=vis_config,
        state_dict=new_ckpt_weights)

    if vis_config.world_model_backend in (TEMPORAL_HYBRID_BACKEND, TEMPORAL_LOOP_BACKEND):
        supplied = {k.removeprefix('bert.') for k in new_ckpt_weights}
        if any(k.startswith('prior.') for k in supplied):
            required = {k for k in visual_model.state_dict() if k.startswith(WORLD_MODEL_PREFIXES)}
            if required - supplied:
                raise ValueError(f'Missing pretrained hybrid parameters: {sorted(required - supplied)}')
    return visual_model
