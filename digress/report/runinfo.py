import os
import platform
import subprocess
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]


def git_info():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=_ROOT, stderr=subprocess.DEVNULL, text=True).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        pass
    # no git binary (e.g. inside the container): read the commit from .git directly; dirty state is unknown
    try:
        head = (_ROOT / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return {"commit": head, "branch": None, "dirty": None}
        ref = head[5:]
        ref_file = _ROOT / ".git" / ref
        if ref_file.exists():
            commit = ref_file.read_text().strip()
        else:  # ref only stored in packed-refs
            packed = (_ROOT / ".git" / "packed-refs").read_text().splitlines()
            commit = next(line.split()[0] for line in packed if line.endswith(" " + ref))
        return {"commit": commit, "branch": ref.removeprefix("refs/heads/"), "dirty": None}
    except (OSError, StopIteration):
        return None


def run_info():
    import rdkit
    import torch_geometric

    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__, "rdkit": rdkit.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(), "git": git_info(),
    }
