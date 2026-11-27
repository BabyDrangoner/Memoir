"""Loop depth is computation depth; source K/V positions remain trajectory time."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from vlnbert.drama_mamba2 import (
    CausalTemporalAttention, DramaMamba2Prior, TEMPORAL_HYBRID_BACKEND,
    TEMPORAL_LOOP_BACKEND,
)
from vlnbert.vlnbert_init import select_drama_weights, get_vlnbert_models


class LoopTemporalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def check_gradients(self, first, second, *, atol=1e-5, rtol=1e-4):
        for (name, a), (other_name, b) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(a.grad, name)
            self.assertTrue(torch.isfinite(a.grad).all(), name)
            self.assertGreater(a.grad.abs().sum().item(), 0, name)
            torch.testing.assert_close(a.grad, b.grad, atol=atol, rtol=rtol, msg=name)

    def test_one_loop_is_legacy_residual_with_identical_rng_and_gradient(self):
        attn = CausalTemporalAttention(24, 3)
        other = copy.deepcopy(attn)
        norm, drop = nn.LayerNorm(24), nn.Dropout(.2)
        norm2 = copy.deepcopy(norm)
        x = torch.randn(2, 7, 24, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        rng = torch.get_rng_state()
        delta, memory = attn(norm(x))
        expected = x + drop(delta)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual, actual_memory = other.refine(x2, norm2, drop, iterations=1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(after, torch.get_rng_state()))
        for a, b in zip(memory, actual_memory):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        expected.backward(upstream)
        actual.backward(upstream)
        torch.testing.assert_close(x.grad, x2.grad, atol=0, rtol=0)
        self.check_gradients(attn, other, atol=0, rtol=0)
        self.check_gradients(norm, norm2, atol=0, rtol=0)

    def test_looped_full_chunk_gradient_and_causality(self):
        attn = CausalTemporalAttention(24, 3)
        other = copy.deepcopy(attn)
        norm = nn.LayerNorm(24)
        norm2 = copy.deepcopy(norm)
        x = torch.randn(2, 21, 24, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        y, cache = attn.refine(x, norm, nn.Identity(), iterations=2)
        parts, chunk_cache = [], None
        for block in x2.split([1, 15, 1, 4], dim=1):
            part, chunk_cache = other.refine(block, norm2, nn.Identity(), chunk_cache, iterations=2)
            parts.append(part)
        yy = torch.cat(parts, 1)
        self.assertEqual(cache[0].shape, (2, 3, 21, 8))
        torch.testing.assert_close(yy, y, atol=2e-6, rtol=2e-5)
        for a, b in zip(cache, chunk_cache):
            torch.testing.assert_close(a, b)
        upstream = torch.randn_like(y)
        y.backward(upstream)
        yy.backward(upstream)
        torch.testing.assert_close(x.grad, x2.grad, atol=3e-6, rtol=3e-5)
        self.check_gradients(attn, other)
        self.check_gradients(norm, norm2)
        with torch.no_grad():
            changed = x.detach().clone()
            changed[:, 6:] += torch.randn_like(changed[:, 6:]) * 7
            z, _ = attn.refine(changed, norm, nn.Identity(), iterations=2)
            torch.testing.assert_close(z[:, :6], y[:, :6], rtol=0, atol=0)
            self.assertFalse(torch.allclose(z[:, 6:], y[:, 6:]))

    def test_memory_is_fixed_across_depth_and_source_cache_is_immutable(self):
        attn, norm = CausalTemporalAttention(24, 3), nn.LayerNorm(24)
        source = torch.randn(2, 5, 24)
        _, history = attn(norm(source))
        saved = tuple(x.clone() for x in history)
        current = torch.randn(2, 3, 24)
        results = [attn.refine(current, norm, nn.Identity(), history, r) for r in (1, 2, 4)]
        for _, memory in results:
            self.assertEqual(memory[0].shape[2], 8)
            for a, b in zip(memory, results[0][1]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(history, saved):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(torch.equal(results[0][0], results[1][0]))

    def make_prior(self, depth):
        return DramaMamba2Prior(d_model=768, n_layers=4, temporal_attention=True,
                                dropout=0, loop_iterations=depth)

    def test_prior_one_loop_matches_hybrid_without_adding_parameters(self):
        old, loop = self.make_prior(None), self.make_prior(1)
        state = old.state_dict()
        state['_loop_iterations'] = torch.tensor(1)
        loop.load_state_dict(state, strict=True)
        self.assertEqual(dict(old.named_parameters()).keys(), dict(loop.named_parameters()).keys())
        x = torch.randn(1, 4, 768, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        language = torch.randn(1, 4, 768)
        mask = torch.tensor([[True, True, True, False]])
        stats, deter, cache = old(x, language, mask)
        stats2, deter2, cache2 = loop(x2, language, mask)
        torch.testing.assert_close(stats, stats2, rtol=0, atol=0)
        torch.testing.assert_close(deter, deter2, rtol=0, atol=0)
        for a, b in zip(cache, cache2):
            for aa, bb in zip(a, b):
                torch.testing.assert_close(aa, bb, rtol=0, atol=0)
        u, v = torch.randn_like(stats), torch.randn_like(deter)
        (stats*u).sum().add((deter*v).sum()).backward()
        (stats2*u).sum().add((deter2*v).sum()).backward()
        torch.testing.assert_close(x.grad, x2.grad, rtol=0, atol=0)
        self.check_gradients(old, loop, rtol=0, atol=0)

    def test_prior_loop_branch_packing_and_single_mamba_update(self):
        prior = self.make_prior(2).eval()
        x, language = torch.randn(2, 3, 768), torch.randn(2, 4, 768)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        with torch.no_grad():
            with patch.object(prior.layers[0], 'forward', wraps=prior.layers[0].forward) as call:
                _, _, history = prior(x, language, mask)
                self.assertEqual(call.call_count, 1)
            saved = [[t.clone() for t in layer] for layer in history]
            branch = [torch.randn(2, 1, 768) for _ in range(2)]
            each = [prior(t, language, mask, history) for t in branch]
            packed = [tuple(torch.cat((t, t)) for t in layer) for layer in history]
            result = prior(torch.cat(branch), language.repeat(2, 1, 1), mask.repeat(2, 1), packed)
            for i in (0, 1):
                torch.testing.assert_close(result[i], torch.cat([v[i] for v in each]), atol=3e-6, rtol=4e-5)
            self.assertEqual(result[2][-1][0].shape, (4, 12, 4, 64))
            for i, layer in enumerate(history):
                for j, tensor in enumerate(layer):
                    torch.testing.assert_close(tensor, saved[i][j], atol=0, rtol=0)
                    torch.testing.assert_close(result[2][i][j], torch.cat([v[2][i][j] for v in each]),
                                               atol=3e-6, rtol=4e-5)
            changed_language = language.masked_fill(~mask.unsqueeze(-1), 99)
            replay = prior(torch.cat((x, branch[0]), 1), changed_language, mask)
            for i in (0, 1):
                torch.testing.assert_close(replay[i], each[0][i], atol=3e-6, rtol=4e-5)

    def test_two_loop_prior_full_and_chunk_backward_are_equivalent(self):
        whole = self.make_prior(2)
        chunked = copy.deepcopy(whole)
        x = torch.randn(1, 5, 768, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        language = torch.randn(1, 4, 768, requires_grad=True)
        language2 = language.detach().clone().requires_grad_()
        mask = torch.tensor([[True, True, True, False]])
        stats, deter, cache = whole(x, language, mask)
        _, _, prefix = chunked(x2[:, :2], language2, mask)
        ss, dd, cc = chunked(x2[:, 2:], language2, mask, prefix)
        torch.testing.assert_close(stats, ss, atol=3e-6, rtol=4e-5)
        torch.testing.assert_close(deter, dd, atol=3e-6, rtol=4e-5)
        self.assertEqual(cc[-1][0].shape[2], 5)
        for layer, other in zip(cache, cc):
            for a, b in zip(layer, other):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=4e-5)
        u, v = torch.randn_like(stats), torch.randn_like(deter)
        ((stats*u).sum() + (deter*v).sum()).backward()
        ((ss*u).sum() + (dd*v).sum()).backward()
        torch.testing.assert_close(x.grad, x2.grad, atol=3e-6, rtol=4e-5)
        torch.testing.assert_close(language.grad, language2.grad, atol=3e-6, rtol=4e-5)
        self.check_gradients(whole, chunked, atol=2e-5, rtol=1e-4)

    def test_fresh_factory_propagates_loop_backend_and_freezes_encoders(self):
        from types import SimpleNamespace
        from transformers import BertConfig
        from transformers.utils import logging
        from vlnbert.drama_mamba2 import freeze_for_world_model_pretraining
        args = SimpleNamespace(bert_ckpt_file=None, tokenizer='bert',
            image_feat_size=512, angle_feat_size=4, obj_feat_size=0,
            num_l_layers=1, num_pano_layers=1, num_x_layers=1, graph_sprels=True,
            fusion='dynamic', fix_lang_embedding=True, fix_pano_embedding=True,
            fix_local_branch=False, freeze_encoders=True, initialize_drama_world=True,
            world_model_backend=TEMPORAL_LOOP_BACKEND, drama_d_model=768,
            drama_layers=4, drama_d_state=16, drama_headdim=64, drama_dropout=0.,
            drama_loop_iterations=2)
        # Small encoders exercise the real HF factory without a dataset or CUDA;
        # the world-model backbone retains the production 4 x 768 dimensions.
        config = BertConfig(hidden_size=24, num_attention_heads=3, intermediate_size=48,
                            vocab_size=97, max_position_embeddings=32)
        old_verbosity = logging.get_verbosity()
        logging.set_verbosity_error()
        try:
            with patch('vlnbert.vlnbert_init.PretrainedConfig.from_pretrained', return_value=config):
                model = get_vlnbert_models(args)
        finally:
            logging.set_verbosity(old_verbosity)
        names = freeze_for_world_model_pretraining(model)
        self.assertEqual(model.config.world_model_backend, TEMPORAL_LOOP_BACKEND)
        self.assertEqual(model.prior.loop_iterations, 2)
        self.assertEqual(model.state_dict()['prior._loop_iterations'].item(), 2)
        self.assertEqual(len(names), 81)
        self.assertNotIn('prior._loop_iterations', names)
        for name in ('embeddings', 'lang_encoder', 'img_embeddings'):
            module = getattr(model, name)
            self.assertFalse(module.training)
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))

    def test_positive_integer_loop_depth(self):
        for value in (0, -1, 2., True, '2'):
            with self.assertRaises(ValueError):
                self.make_prior(value)
        with self.assertRaises(ValueError):
            DramaMamba2Prior(loop_iterations=2)
        a = CausalTemporalAttention(24, 3)
        with self.assertRaises(ValueError):
            a.refine(torch.randn(1, 2, 24), nn.Identity(), nn.Identity(), iterations=0)

    def test_checkpoint_identity_rejects_unmarked_or_wrong_backbone_and_depth(self):
        prefix = 'bert.prior.layers.'
        common = {prefix + str(i) + '.in_proj.weight': None for i in range(3)}
        mixed = {**common, **{prefix + '3.' + n: None for n in
                             ('qkv.weight', 'out_proj.weight', 'relative_bias.weight')}}
        marked = {**mixed, 'bert.prior._loop_iterations': torch.tensor(2)}
        self.assertIs(select_drama_weights(marked, TEMPORAL_LOOP_BACKEND, False, 2), marked)
        for weights, backend, depth in ((mixed, TEMPORAL_LOOP_BACKEND, 2),
                                       (marked, TEMPORAL_HYBRID_BACKEND, None),
                                       (marked, 'drama_mamba2', None),
                                       (marked, TEMPORAL_LOOP_BACKEND, 3)):
            with self.assertRaises(ValueError):
                select_drama_weights(weights, backend, True, depth)
        for marker in (None, torch.tensor(2.), torch.tensor([2]), torch.tensor(True)):
            with self.assertRaises(ValueError):
                select_drama_weights({**mixed, 'bert.prior._loop_iterations': marker},
                                     TEMPORAL_LOOP_BACKEND, False, 2)
        author = {'bert.prior.transformer.weight': None, 'bert.posterior.weight': None,
                  'bert.embeddings.weight': None}
        self.assertEqual(select_drama_weights(author, TEMPORAL_LOOP_BACKEND, True, 2),
                         {'bert.embeddings.weight': None})
        prior = self.make_prior(2)
        state = prior.state_dict()
        state['_loop_iterations'] = torch.tensor(3)
        with self.assertRaisesRegex(RuntimeError, 'loop depth differs'):
            prior.load_state_dict(state, strict=True)
        del state['_loop_iterations']
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            prior.load_state_dict(state, strict=True)


if __name__ == '__main__':
    unittest.main()
