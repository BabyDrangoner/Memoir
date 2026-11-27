"""Loop backend identity, true WM-loss training, and strict resume audits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from test_drama_pretrain import TinyWorld, batch
import drama_pretrain as pre
from colab_full_architecture import ARCH, MAMBA2, TEMPORAL_HYBRID, TEMPORAL_LOOP, architecture
from colab_full_pretrain import audit_gradients, backward_update, frozen_fingerprints, restore_weights
from vlnbert.drama_mamba2 import DramaMamba2Prior, freeze_for_world_model_pretraining


class LoopPretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        pre.seed_all(77)

    def test_architecture_identity_and_loop_configuration_rejections(self):
        base = architecture(ARCH, TEMPORAL_HYBRID)
        loop = architecture(dict(ARCH, loop_iterations=2), TEMPORAL_LOOP)
        self.assertNotEqual(loop, base)
        self.assertEqual(loop['loop_iterations'], 2)
        self.assertEqual(loop['loop_memory'], 'fixed_source_kv_per_time_step')
        self.assertFalse(loop['loop_ffn'])
        for invalid in (None, 0, -1, True, 1.5, '2'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'positive integer'):
                architecture(dict(ARCH, loop_iterations=invalid), TEMPORAL_LOOP)
        for backend in (MAMBA2, TEMPORAL_HYBRID):
            with self.assertRaisesRegex(ValueError, 'only'):
                architecture(dict(ARCH, loop_iterations=2), backend)
        with self.assertRaises(ValueError):
            architecture(dict(ARCH, d_model=512, loop_iterations=2), TEMPORAL_LOOP)

    def test_model_arguments_and_smoke_schedule_do_not_claim_completion(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/loop_pretrain_smoke.json').read_text())
        cfg['init_checkpoint'] = '/not-opened/base.pt'
        args = pre.model_args(cfg)
        self.assertEqual(args.world_model_backend, TEMPORAL_LOOP)
        self.assertEqual(args.drama_loop_iterations, 2)
        self.assertTrue(args.freeze_encoders)
        self.assertTrue(args.initialize_drama_world)
        self.assertEqual((cfg['steps'], cfg['warmup_steps'], cfg['batch_size'], cfg['microbatch']),
                         (5000, 100, 32, 2))
        self.assertEqual(cfg['overshoot_horizon'], 5)
        self.assertEqual(pre.scheduled_lr(100, cfg), 5e-5)
        self.assertGreater(pre.scheduled_lr(102, cfg), 4.9e-5)
        for invalid in (None, True, 0, 1.5):
            with self.assertRaises(ValueError):
                pre.model_args(dict(cfg, loop_iterations=invalid))
        old = dict(cfg, world_model_backend=TEMPORAL_HYBRID)
        with self.assertRaises(ValueError):
            pre.model_args(old)
        del old['loop_iterations']
        self.assertFalse(hasattr(pre.model_args(old), 'drama_loop_iterations'))

    def test_frozen_hash_includes_buffers_and_detects_mutation(self):
        model = nn.Linear(3, 2)
        model.register_buffer('_loop_iterations', torch.tensor(2))
        names = set(model.state_dict())
        expected = frozen_fingerprints(model, names)
        self.assertEqual(expected, frozen_fingerprints(model, names))
        self.assertIn('_loop_iterations', expected)
        with torch.no_grad():
            model._loop_iterations.fill_(3)
        actual = frozen_fingerprints(model, names)
        self.assertNotEqual(expected['_loop_iterations'], actual['_loop_iterations'])
        self.assertEqual(expected['weight'], actual['weight'])

    def test_gradient_audit_rejects_frozen_or_nonfinite_gradient(self):
        model = nn.ModuleDict({'frozen': nn.Linear(2, 2), 'trainable': nn.Linear(2, 2)})
        model['frozen'].requires_grad_(False)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        model['trainable'](torch.ones(1, 2)).sum().backward()
        report = audit_gradients(model, trainable)
        self.assertEqual(report['zero_gradient_names'], [])
        model['frozen'].weight.grad = torch.zeros_like(model['frozen'].weight)
        with self.assertRaisesRegex(RuntimeError, 'Frozen parameter'):
            audit_gradients(model, trainable)
        model['frozen'].weight.grad = None
        model['trainable'].weight.grad[0, 0] = float('nan')
        with self.assertRaisesRegex(FloatingPointError, 'nonfinite'):
            audit_gradients(model, trainable)

    @staticmethod
    def loop_world():
        model = TinyWorld()
        # Keep the actual production 3xMamba+loop-attention core; small heads/data
        # reduce test cost without substituting fake dynamics or a fake loss.
        model.prior = DramaMamba2Prior(2, False, 8, 6, d_model=768, n_layers=4,
                                      d_state=16, headdim=64, dropout=.1,
                                      temporal_attention=True, loop_iterations=2)
        freeze_for_world_model_pretraining(model)
        return model

    def test_actual_loop_wm_loss_full_state_restore_and_next_update(self):
        model = self.loop_world()
        cfg = dict(lr=5e-5, betas=[.9, .98], weight_decay=.01, adam_eps=1e-6,
                   grad_clip=5., warmup_steps=100, steps=5000)
        optimizer = pre.build_optimizer(model, cfg)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        frozen_names = model.state_dict().keys() - trainable.keys()
        baseline = frozen_fingerprints(model, frozen_names)
        data = batch()
        sampler = pre.Sampler(19)

        def update(m, opt, step):
            opt.zero_grad(set_to_none=True)
            value, parts = backward_update(m, data, 1, 3, {})
            report = audit_gradients(m, {n: p for n, p in m.named_parameters() if p.requires_grad})
            self.assertEqual(set(parts), {'kl', 'overshoot_kl', 'visual_nce', 'reward_mse'})
            self.assertGreater(value, 0)
            self.assertFalse(report['zero_gradient_names'])
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
            for group in opt.param_groups:
                group['lr'] = pre.scheduled_lr(step, cfg)
            opt.step()
            return value

        update(model, optimizer, 1)
        self.assertEqual(frozen_fingerprints(model, frozen_names), baseline)
        sampler.take(7)
        state = dict(trainable_state={n: p.detach().clone() for n, p in trainable.items()},
                     optimizer=copy.deepcopy(optimizer.state_dict()), sampler=sampler.state(),
                     rng=pre.rng_state(), scheduler=pre.scheduler_state(1, cfg), step=1)
        with tempfile.TemporaryDirectory() as tmp:
            frozen = Path(tmp) / 'frozen-base.pt'
            torch.save({n: model.state_dict()[n].detach().clone() for n in frozen_names}, frozen)
            expected_indices = sampler.take(7)
            expected_loss = update(model, optimizer, 2)
            expected_parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
            restored = self.loop_world()
            restored_trainable = {n: p for n, p in restored.named_parameters() if p.requires_grad}
            restore_weights(restored, restored_trainable, state, frozen)
            restored_opt = pre.build_optimizer(restored, cfg)
            restored_opt.load_state_dict(state['optimizer'])
            pre.validate_scheduler(state['scheduler'], state['step'], cfg, restored_opt)
            pre.set_rng(state['rng'])
            sampler2 = pre.Sampler(19, state['sampler'])
            self.assertEqual(sampler2.take(7), expected_indices)
            self.assertEqual(update(restored, restored_opt, 2), expected_loss)
            for name, parameter in restored.named_parameters():
                torch.testing.assert_close(parameter, expected_parameters[name], rtol=0, atol=0, msg=name)
            self.assertEqual(frozen_fingerprints(restored, frozen_names), baseline)
            broken = dict(state, trainable_state={})
            with self.assertRaisesRegex(RuntimeError, 'partition'):
                restore_weights(restored, restored_trainable, broken, frozen)


if __name__ == '__main__':
    unittest.main()
