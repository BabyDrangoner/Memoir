import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_drama_pretrain import TinyWorld, batch
import torch
from colab_full_pretrain import TrainingRows, backward_update
import drama_pretrain as pre


class FullDataTests(unittest.TestCase):
    def test_full_negative_pool_and_uneven_trajectory_normalizers(self):
        torch.set_num_threads(1)
        torch.manual_seed(177)
        full = TinyWorld().eval().double()
        micro = copy.deepcopy(full)
        data = {key: (val.double() if val.is_floating_point() else val) for key, val in batch().items()}
        # Couple stochastic latent draws between different batch layouts to isolate the objective.
        with patch.object(torch.distributions.Normal, "rsample", lambda dist: dist.mean):
            loss, pieces = pre.wm_loss(full, data, 3)
            loss.backward()
            actual, parts = backward_update(micro, data, 1, 3, {})
        self.assertAlmostEqual(loss.item(), actual, places=6)
        for key, val in pieces.items():
            self.assertAlmostEqual(val.item(), parts[key], places=6)
        for (name, p), (_, other) in zip(full.named_parameters(), micro.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(other.grad, p.grad, rtol=3e-5, atol=3e-6, msg=name)

    def test_every_instruction_and_jsonl_line_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, augmented = Path(tmp)/"human.json", Path(tmp)/"aug.jsonl"
            human.write_text(json.dumps([{"path_id": 1, "scan": "a", "heading": 0, "path": ["a","b"],
                                          "instructions": ["first","second"], "instr_encodings": [[1],[2]]}]))
            augmented.write_text('{"instr_id":"1_0","instr_encoding":[3]}\n{"instr_id":"1_0","instr_encoding":[4]}\n')
            rows = TrainingRows(human, augmented)
            try:
                self.assertEqual(len(rows), 4)
                self.assertEqual([rows[i]["instr_encoding"] for i in [2, 0, 3, 1]], [[3], [1], [4], [2]])
            finally:
                rows.augmented.close()


class RuntimeLocalStorageTests(unittest.TestCase):
    root = "19 1 0:42 / / rw,relatime - overlay overlay rw\n"
    drive = "41 19 0:43 / /content/drive rw - fuse.drive drive rw\n"
    boot = "44b47729-6a30-4e19-971a-089a8d0c5519"

    def proc(self, mounts, boot=None):
        def read(path):
            if str(path) == "/proc/self/mountinfo":
                return mounts
            if str(path) == "/proc/sys/kernel/random/boot_id":
                return boot or self.boot
            raise AssertionError(path)
        return patch.object(Path, "read_text", autospec=True, side_effect=read)

    def test_runtime_identity_and_strict_mount_contract(self):
        from drama_mount_guard import backup_mount_identity, checkpoint_mount_contract, validate_backup_root
        dest = Path("/content/project/experiment/checkpoints")
        with patch.object(Path, "is_file", return_value=True), self.proc(self.root + self.drive):
            identity = backup_mount_identity(dest, backend="runtime-local")
            self.assertEqual(identity["filesystem"], "overlay")
            self.assertEqual(identity["boot_id"], self.boot)
            self.assertEqual(checkpoint_mount_contract(identity)["mount_id"], "19")
            validate_backup_root(dest, backend="runtime-local", expected_identity=identity)
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                backup_mount_identity(dest, backend="drive")
            with patch.object(Path, "is_dir", return_value=True), self.assertRaisesRegex(RuntimeError, "persistent filesystem"):
                backup_mount_identity(dest, backend="filesystem")
        with patch.object(Path, "is_file", return_value=True), self.proc(self.root, "7ea74db3-c904-44b8-9f88-99b7ed9622fb"):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                validate_backup_root(dest, backend="runtime-local", expected_identity=identity)

    def test_runtime_store_excludes_drive_submounts_and_other_disks(self):
        from drama_mount_guard import backup_mount_identity
        mount = "28 19 8:1 / /content/mounted rw - ext4 /dev/sda rw\n"
        with patch.object(Path, "is_file", return_value=True), self.proc(self.root + self.drive + mount):
            for value in ("/content/drive/test", "/content/mounted/weights", "/tmp/test"):
                with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Colab /content"):
                    backup_mount_identity(Path(value), backend="runtime-local")
        with patch.object(Path, "is_file", return_value=True), self.proc(self.root.replace("overlay overlay", "tmpfs tmpfs")):
            with self.assertRaisesRegex(RuntimeError, "writable local disk"):
                backup_mount_identity(Path("/content/experiment"), backend="runtime-local")
