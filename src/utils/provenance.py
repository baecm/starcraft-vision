"""Which code, and which machine, produced an artefact.

Training records this next to its weights. Inference did not, so a prediction
file could not say what produced it - and predictions are exactly what gets
split across machines when a sweep is too slow on one. A batch of ablations was
once launched across workstations whose checkouts were at different commits and
three of them silently reproduced the previous round; that was caught by
checksumming files afterwards, which only works if you already suspect it.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Optional


def git_state() -> dict:
    """HEAD commit and whether the working tree is dirty, or why we cannot tell.

    The container mounts only src/, so there is no .git under /workspace and
    asking git directly fails. The Makefile reads the host checkout at launch
    and passes GIT_COMMIT / GIT_DIRTY through the environment; the subprocess
    path is the fallback for running outside the container.
    """
    env_commit = os.environ.get("GIT_COMMIT", "").strip()
    if env_commit:
        return {
            "commit": env_commit,
            "dirty": os.environ.get("GIT_DIRTY", "").strip() == "1",
            "source": "host env",
        }

    def _run(args):
        return subprocess.check_output(
            args,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        ).decode().strip()

    try:
        return {
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "subject": _run(["git", "log", "-1", "--format=%s"]),
            "committed_at": _run(["git", "log", "-1", "--format=%cI"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
            "source": "subprocess",
        }
    except Exception as e:
        return {
            "commit": None,
            "error": f"{type(e).__name__}: {e}",
            "hint": "GIT_COMMIT was not in the environment and there is no .git "
                    "here; launch via the Makefile, which exports it from the "
                    "host checkout.",
        }


def host_tag() -> Optional[str]:
    """Which workstation and which physical GPU, e.g. worker07:0.

    Neither half is discoverable inside the container: its hostname is the
    container id, and NVIDIA_VISIBLE_DEVICES=2 exposes that one card as cuda:0,
    so torch reports device 0 whichever card is in use. The Makefile exports
    HOST_NAME for the same reason it exports GIT_COMMIT.
    """
    host = (os.environ.get("HOST_NAME") or "").strip()
    if not host:
        return None
    devices = (os.environ.get("NVIDIA_VISIBLE_DEVICES") or "").strip()
    # "all"/"none"/"void" are the CDI keywords, not card indices
    if not devices or devices.lower() in ("all", "none", "void"):
        return host
    return f"{host}:{devices}"


def write_provenance(path: str, record: dict) -> dict:
    """Write `record` plus the code and machine it ran on. Returns what it wrote."""
    full = dict(record)
    full.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    full["git"] = git_state()
    full["host"] = host_tag()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, sort_keys=True)
    return full
