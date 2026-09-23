"""Which trained runs are still comparable to the current code.

A checkpoint is only usable if the code that produced it still behaves the way
the current checkout does. Four commits (ed86508, 11b95d7, 5fdf73f, 544359a)
changed the render, head initialisation, peak extraction and loss domains, and
every checkpoint trained before them stopped being comparable to one trained
after - without changing name, size, or anything else visible in a directory
listing. A previous round lost three ablations to exactly this: they were
launched from stale checkouts on other machines and silently reproduced the
earlier run, which was only caught afterwards by checksumming predictions.

`train.py` writes `run_provenance.json` next to the weights, recording the
commit and every knob that changes model behaviour. This script reads those
files and answers the two questions a run list has to answer:

  - was this run trained before a change that invalidates it
  - do two runs differ in a knob that makes them incomparable regardless of
    commit (a stride-2 run is not another seed of a stride-4 run)

Runs are grouped by that knob signature, so incomparable runs cannot land in
adjacent rows of one table by accident. A run with no provenance file predates
the feature and is reported as UNKNOWN rather than assumed good.

The containers do not mount `.git` (see infra/docker-compose.yml), so ancestry
cannot be resolved in-container. The Makefile target resolves it on the host
and passes the result as --ok-commits-file; run the script directly on a host
checkout and --since works on its own.

Usage:
  make run-registry ARGS="--since ed86508"
  python3 scripts/run_registry.py --since ed86508 --out docs/run_registry.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Set

# Knobs that change what the model computes. Two runs differing in any of these
# are different models, not different seeds of one.
SIGNATURE_KNOBS = (
    "model_class",
    "down_ratio",
    "k_max",
    "conf_threshold",
    "render_sigma",
    "u_observers",
    "soft_center_radius",
    "peak_border_margin",
    "trainable_layers",
    "head_conv",
    "dense_positives",
)


def _git(args: List[str], repo: str) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args, cwd=repo,
                             capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _epochs(pred_root: str, run: str) -> List[str]:
    d = os.path.join(pred_root, run)
    if not os.path.isdir(d):
        return []
    return sorted(e for e in os.listdir(d) if e.startswith("model_"))


def _signature(rec: Dict) -> str:
    parts = [f"{k}={rec[k]}" for k in SIGNATURE_KNOBS
             if rec.get(k) is not None]
    return " ".join(parts) or "(no knobs recorded)"


def collect(models_root: str, pred_root: str) -> List[Dict]:
    runs = []
    if not os.path.isdir(models_root):
        return runs
    for name in sorted(os.listdir(models_root)):
        d = os.path.join(models_root, name)
        if not os.path.isdir(d):
            continue
        ckpts = sorted(f for f in os.listdir(d) if f.endswith(".pth"))
        if not ckpts:
            continue
        rec = {}
        path = os.path.join(d, "run_provenance.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except (OSError, ValueError):
                rec = {}
        runs.append({
            "run": name,
            "rec": rec,
            "checkpoints": ckpts,
            "predictions": _epochs(pred_root, name),
        })
    return runs


def status_of(rec: Dict, ok_commits: Optional[Set[str]]) -> str:
    if not rec:
        return "UNKNOWN (no provenance)"
    git = rec.get("git") or {}
    commit = git.get("commit")
    if not commit:
        return "UNKNOWN (no commit)"
    if git.get("dirty"):
        return "NOT REPRODUCIBLE (dirty tree)"
    if ok_commits is None:
        return "recorded"
    return "ok" if commit in ok_commits else "STALE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--pred-root", default="predictions")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--since", default=None,
                    help="boundary commit; runs not trained in <since>..HEAD "
                         "are STALE. Needs .git, so use --ok-commits-file in a "
                         "container")
    ap.add_argument("--ok-commits-file", default=None,
                    help="file of commits considered current, one per line, "
                         "as produced on the host by git rev-list <since>..HEAD")
    ap.add_argument("--out", default=None, help="also write markdown here")
    args = ap.parse_args()

    ok: Optional[Set[str]] = None
    boundary_note = None
    if args.ok_commits_file:
        try:
            with open(args.ok_commits_file, encoding="utf-8") as fh:
                ok = {line.strip() for line in fh if line.strip()}
            boundary_note = (f"{len(ok)} commits considered current "
                             f"(from {args.ok_commits_file})")
        except OSError as e:
            print(f"could not read {args.ok_commits_file}: {e}")
            return 1
    elif args.since:
        listed = _git(["rev-list", f"{args.since}..HEAD"], args.repo)
        head = _git(["rev-parse", "HEAD"], args.repo)
        if listed is None or head is None:
            print("git is unavailable here (the containers do not mount .git);"
                  " pass --ok-commits-file instead")
            return 1
        ok = {c for c in listed.splitlines() if c}
        ok.add(head)
        boundary_note = f"trained after {args.since} ({len(ok)} commits)"

    runs = collect(args.models_root, args.pred_root)
    if not runs:
        print(f"no runs with checkpoints under {args.models_root}")
        return 1

    lines: List[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"# Run registry ({len(runs)} runs with checkpoints)")
    emit()
    head = _git(["rev-parse", "HEAD"], args.repo)
    if head:
        emit(f"Current HEAD: `{head[:9]}`")
    if boundary_note:
        emit(f"Boundary: {boundary_note}. STALE = trained before it.")
    emit()

    groups = defaultdict(list)
    for r in runs:
        groups[_signature(r["rec"])].append(r)

    for sig, members in sorted(groups.items()):
        emit(f"## {sig}")
        emit()
        emit("| run | commit | status | trained | ckpts | predictions |")
        emit("| --- | --- | --- | --- | --- | --- |")
        for r in sorted(members, key=lambda x: x["rec"].get("started_at", "")):
            git = r["rec"].get("git") or {}
            commit = git.get("commit") or ""
            short = commit[:9] if commit else "-"
            preds = ", ".join(e[len("model_"):] for e in r["predictions"]) or "-"
            emit(f"| `{r['run']}` | `{short}` | {status_of(r['rec'], ok)} | "
                 f"{r['rec'].get('started_at', '-')} | "
                 f"{len(r['checkpoints'])} | {preds} |")
        emit()

    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
