"""Vendored production mount checks; no local-test bypass exposed by launcher."""
from pathlib import Path
import re
from uuid import UUID
from typing import Callable
from drama_checkpoint_store import CheckpointStore

STORAGE_BACKENDS = ("drive", "filesystem", "runtime-local")


def checkpoint_mount_contract(mount: dict) -> dict:
    keys = ("backend", "mount_root", "filesystem", "source")
    if mount["backend"] == "runtime-local":
        keys += ("boot_id", "mount_id", "device", "mount_point")
    return {key: mount[key] for key in keys}


def backup_mount_identity(path: Path, *, backend: str = "drive", allow_local: bool = False) -> dict:
    """Validate an explicit backup backend and identify its deepest Linux mount.

    Filesystem backup protects process/restart recovery. A directory on the same
    disk is not protection against disk failure. No directory is created here.
    """
    if backend not in STORAGE_BACKENDS:
        raise ValueError(f"Unknown checkpoint storage backend {backend!r}")
    if allow_local:
        if backend != "drive":
            raise ValueError(f"{backend} backup cannot use the test-only local bypass")
        return {"backend": "test-only", "verification": "local-filesystem-test"}
    path = Path(path).expanduser()
    if backend == "filesystem" and not path.is_dir():
        raise RuntimeError(f"filesystem backup root must be an existing directory: {path}")
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise RuntimeError("backup mount verification requires Linux /proc/self/mountinfo")
    target = path.resolve()
    mounts = []
    for line in mountinfo.read_text().splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        fields, details = left.split(), right.split()
        if len(fields) < 6 or len(details) < 2:
            continue
        # mountinfo escapes spaces, tabs, newlines and literal backslashes with
        # octal sequences in paths AND source names (e.g. Windows G:\134).
        decode = lambda value: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)
        mount = Path(decode(fields[4]))
        if target == mount or target.is_relative_to(mount):
            mounts.append((len(mount.parts), {"backend": backend, "mount_point": str(mount),
                "mount_root": decode(fields[3]), "filesystem": details[0], "source": decode(details[1]),
                "device": fields[2], "mount_id": fields[0], "mount_options": fields[5].split(",")}))
    deepest = max((depth for depth, _ in mounts), default=-1)
    matches = [item for depth, item in mounts if depth == deepest]
    if len(matches) > 1:
        raise RuntimeError("ambiguous stacked backup mounts at the same path; refusing to select a covered mount")
    identity = matches[0] if matches else None
    if backend == "runtime-local":
        content = Path("/content")
        if (not target.is_relative_to(content) or target.is_relative_to(content / "drive")
                or identity is None or not content.is_relative_to(identity["mount_point"])):
            raise RuntimeError(f"runtime-local checkpoints require the Colab /content runtime disk: {target}")
        if (identity["filesystem"] not in {"overlay", "fuse.overlayfs", "fuse-overlayfs", "ext4", "xfs", "btrfs"}
                or "ro" in identity["mount_options"]):
            raise RuntimeError(f"runtime-local checkpoints require a writable local disk: {target}")
        identity["boot_id"] = str(UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
        identity["verification"] = "ephemeral-runtime-mount-readback-sha256"
        return identity
    if backend == "drive" and (identity is None or identity["filesystem"] not in {"fuse.drivefs", "fuse.drive"}):
        raise RuntimeError(f"Google Drive is not mounted at backup destination: {path}")
    if backend == "filesystem":
        unsuitable = {"tmpfs", "ramfs", "rootfs", "overlay", "fuse.overlayfs", "fuse-overlayfs",
                      "unionfs", "fuse.unionfs", "aufs", "devtmpfs", "proc", "sysfs",
                      "cgroup", "cgroup2", "squashfs", "debugfs", "tracefs", "securityfs", "fusectl", "mqueue"}
        if identity is None or identity["filesystem"] in unsuitable or identity["source"].startswith(("/dev/ram", "/dev/zram")):
            raise RuntimeError(f"filesystem backup needs a persistent filesystem; mounted type is {identity['filesystem'] if identity else 'unknown'}: {path}")
        if "ro" in identity["mount_options"]:
            raise RuntimeError(f"filesystem backup mount is read-only: {path}")
    identity["verification"] = "drive-mount-readback-sha256" if backend == "drive" else "filesystem-mount-readback-sha256"
    return identity

def validate_backup_root(path: Path, *, backend: str = "drive", allow_local: bool = False,
                         expected_identity: dict | None = None) -> str:
    """Default to Drive; detect a lost/remounted filesystem before any new write."""
    identity = backup_mount_identity(path, backend=backend, allow_local=allow_local)
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError("backup mount identity changed; refusing fallback to a parent or replacement mount")
    return identity["verification"]

def validate_separate_roots(local: Path, backup: Path) -> None:
    local, backup = Path(local).resolve(), Path(backup).resolve()
    if local.is_relative_to(backup) or backup.is_relative_to(local):
        raise ValueError("local and backup roots must be separate, non-nested directories")

class MountedCheckpointStore(CheckpointStore):
    """Check the selected mount again before publishing pointers or pruning."""
    def __init__(self, *args, backup_check: Callable[[], object], **kwargs):
        self.backup_check = backup_check
        self.backup_check()
        super().__init__(*args, **kwargs)

    def _copy_snapshot(self, source, root):
        self.backup_check()
        result = super()._copy_snapshot(source, root)
        self.backup_check()
        return result

    def _update_pointers(self, root, directory, *, is_best):
        self.backup_check()
        return super()._update_pointers(root, directory, is_best=is_best)

    def _prune(self):
        self.backup_check()
        return super()._prune()
