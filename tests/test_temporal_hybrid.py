import copy
import unittest
from unittest.mock import patch

import torch

from vlnbert.drama_mamba2 import (
    CausalTemporalAttention, DramaMamba2Prior, FunctionalMamba2, TEMPORAL_HYBRID_BACKEND,
)
from vlnbert.vlnbert_init import select_drama_weights


class TemporalHybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)

    def test_relative_lags_causal_mask_and_gradient_equivalence(self):
        whole = CausalTemporalAttention(24, 3)
        chunked = copy.deepcopy(whole)
        lags = torch.tensor([0, 1, 15, 16, 17, 32, 64, 128, 1024])
        self.assertEqual(whole._relative_buckets(lags).tolist(), [0, 1, 15, 16, 16, 21, 26, 31, 31])

        x = torch.randn(2, 21, 24, requires_grad=True)
        y, (k, v) = whole(x)
        x2 = x.detach().clone().requires_grad_()
        cache = None
        parts = []
        for block in x2.split([1, 15, 1, 4], dim=1):
            part, cache = chunked(block, cache)
            parts.append(part)
        y2 = torch.cat(parts, dim=1)
        torch.testing.assert_close(y2, y)
        torch.testing.assert_close(cache[0], k)
        torch.testing.assert_close(cache[1], v)
        upstream = torch.randn_like(y)
        y.backward(upstream)
        y2.backward(upstream)
        torch.testing.assert_close(x.grad, x2.grad, atol=2e-6, rtol=2e-5)
        for (name, param), (_, other) in zip(whole.named_parameters(), chunked.named_parameters()):
            torch.testing.assert_close(param.grad, other.grad, atol=6e-6, rtol=2e-5, msg=name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)

        with torch.no_grad():
            changed = x.clone()
            changed[:, 6:] += 10
            changed_y, _ = whole(changed)
            torch.testing.assert_close(y[:, :6], changed_y[:, :6], rtol=0, atol=0)
            self.assertFalse(torch.allclose(y[:, 6:], changed_y[:, 6:]))

    def test_prior_cache_imagination_batch_and_instruction_mask(self):
        prior = DramaMamba2Prior(d_model=768, n_layers=4, temporal_attention=True, dropout=0).eval()
        self.assertEqual([type(layer) for layer in prior.layers], [FunctionalMamba2] * 3 + [CausalTemporalAttention])
        self.assertEqual((prior.layers[3].n_heads, prior.layers[3].head_dim), (12, 64))
        batch = 2
        history = torch.randn(batch, 3, 768)
        language = torch.randn(batch, 4, 768)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        branches = [torch.randn(batch, 1, 768) for _ in range(2)]
        with torch.no_grad(), patch.object(prior.language_k, 'forward', wraps=prior.language_k.forward) as k_proj:
            _, _, history_cache = prior(history, language, mask)
            self.assertEqual(history_cache[-1][0].shape, (batch, 12, 3, 64))
            snapshot = [[tensor.clone() for tensor in layer] for layer in history_cache]
            independent = [prior(x, language, mask, history_cache) for x in branches]
            packed_cache = [tuple(torch.cat((tensor, tensor)) for tensor in layer) for layer in history_cache]
            stats, deter, result_cache = prior(torch.cat(branches), language.repeat(2, 1, 1),
                                               mask.repeat(2, 1), packed_cache)
            self.assertEqual(k_proj.call_count, 1)
            self.assertEqual(stats.shape, (2 * batch, 192))
            self.assertEqual(deter.shape, (2 * batch, 672))
            for packed, idx in [(stats, 0), (deter, 1)]:
                torch.testing.assert_close(packed, torch.cat([one[idx] for one in independent]), atol=2e-6, rtol=3e-5)
            for idx, layer in enumerate(history_cache):
                for slot, tensor in enumerate(layer):
                    torch.testing.assert_close(tensor, snapshot[idx][slot], rtol=0, atol=0)
                    expected = torch.cat([one[2][idx][slot] for one in independent])
                    torch.testing.assert_close(result_cache[idx][slot], expected, atol=2e-6, rtol=3e-5)
            modified_language = language.masked_fill(~mask.unsqueeze(-1), 99)
            replay = prior(torch.cat((history, branches[0]), dim=1), modified_language, mask)
            torch.testing.assert_close(replay[0], independent[0][0], atol=2e-6, rtol=3e-5)
            torch.testing.assert_close(replay[1], independent[0][1], atol=2e-6, rtol=3e-5)

    def test_checkpoint_backend_identity_and_fresh_initialization(self):
        mamba = 'drama_mamba2'
        hybrid = TEMPORAL_HYBRID_BACKEND
        prefix = 'bert.prior.layers.'
        common = {prefix + f'{i}.in_proj.weight': None for i in range(3)}
        pure = {**common, prefix + '3.in_proj.weight': None}
        mixed = {**common, **{prefix + '3.' + name: None for name in (
            'qkv.weight', 'out_proj.weight', 'relative_bias.weight')}}
        for weights, expected, wrong in [(pure, mamba, hybrid), (mixed, hybrid, mamba)]:
            self.assertIs(select_drama_weights(weights, expected, False), weights)
            with self.assertRaises(ValueError):
                select_drama_weights(weights, wrong, True)
        broken = mixed.copy()
        del broken[prefix + '3.qkv.weight']
        with self.assertRaises(ValueError):
            select_drama_weights(broken, hybrid, False)
        tssm = {'bert.prior.transformer.weight': None, 'bert.posterior.weight': None, 'bert.embeddings.weight': None}
        with self.assertRaises(ValueError):
            select_drama_weights(tssm, hybrid, False)
        self.assertEqual(select_drama_weights(tssm, hybrid, True), {'bert.embeddings.weight': None})
        with self.assertRaises(ValueError):
            DramaMamba2Prior(d_model=512, n_layers=2, temporal_attention=True)
