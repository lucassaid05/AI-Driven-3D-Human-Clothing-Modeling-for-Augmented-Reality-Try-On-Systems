#!/usr/bin/env python3
"""
Wrapper for HumanSplat (https://humansplat.github.io/) — does not vendor the project.

After cloning the official repository and installing its environment, point
HUMANSPLAT_REPO at the repo root and optionally set:

  HUMANSPLAT_CMD   Shell command to run inference (default: see below)
  HUMANSPLAT_ARGS  Extra args appended after input/output flags (optional)

If HUMANSPLAT_CMD is unset, we try, in order:
  python infer.py --help   (many research repos use this name)
  python demo.py --help

This script only forwards argv[1:] to the subprocess when using --pass-through.

Examples:
  export HUMANSPLAT_REPO=/path/to/HumanSplat
  python pipeline/run_humansplat.py --input person_crop.png --out human.ply

  # Custom driver inside their repo:
  export HUMANSPLAT_CMD="python scripts/recon.py"
  python pipeline/run_humansplat.py --pass-through -- --config cfg.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _default_cmd(repo: Path) -> list[str]:
    """Pick a plausible entrypoint; override with HUMANSPLAT_CMD if needed."""
    exe = shutil.which("python") or sys.executable
    for name in ("infer.py", "demo.py", "main.py", "run.py"):
        p = repo / name
        if p.is_file():
            return [exe, str(p)]
    return [exe, str(repo / "infer.py")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=Path, help="Person crop image (RGB)")
    p.add_argument("--out", "-o", type=Path, help="Output path for 3DGS / mesh (repo-specific)")
    p.add_argument(
        "--pass-through",
        action="store_true",
        help="Forward all following args to HUMANSPLAT_CMD (use -- before their flags)",
    )
    args, rest = p.parse_known_args()

    repo_s = os.environ.get("HUMANSPLAT_REPO", "").strip()
    if not repo_s:
        print(
            "[HumanSplat] HUMANSPLAT_REPO is not set.\n"
            "  1. Clone the code from https://humansplat.github.io/\n"
            "  2. Follow their README for checkpoints and dependencies\n"
            "  3. export HUMANSPLAT_REPO=/absolute/path/to/clone\n"
            "  4. Optionally export HUMANSPLAT_CMD='python their_entry.py'\n"
            "  5. Re-run this wrapper or load the exported .ply in pages/splat-viewer.html\n",
            file=sys.stderr,
        )
        sys.exit(2)

    repo = Path(repo_s).resolve()
    if not repo.is_dir():
        print(f"[HumanSplat] Not a directory: {repo}", file=sys.stderr)
        sys.exit(2)

    cmd_env = os.environ.get("HUMANSPLAT_CMD", "").strip()
    if args.pass_through:
        if not cmd_env:
            print("[HumanSplat] --pass-through requires HUMANSPLAT_CMD", file=sys.stderr)
            sys.exit(2)
        import shlex

        base = shlex.split(cmd_env)
        full = base + rest
        print(f"[HumanSplat] cwd={repo}\n[HumanSplat] exec: {full}")
        r = subprocess.run(full, cwd=str(repo))
        sys.exit(r.returncode)

    if not args.input or not args.out:
        p.error("--input and --out are required unless using --pass-through")

    if cmd_env:
        import shlex

        base = shlex.split(cmd_env)
    else:
        base = _default_cmd(repo)

    # Conventional flags (repos differ — user sets HUMANSPLAT_CMD when these are wrong)
    extra = os.environ.get("HUMANSPLAT_ARGS", "").strip()
    import shlex

    tail = shlex.split(extra) if extra else []
    full = base + ["--input", str(args.input.resolve()), "--output", str(args.out.resolve())] + tail
    print(f"[HumanSplat] cwd={repo}\n[HumanSplat] exec: {full}")
    print(
        "[HumanSplat] If this fails, set HUMANSPLAT_CMD to the exact command from the upstream README.",
        file=sys.stderr,
    )
    r = subprocess.run(full, cwd=str(repo))
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"[HumanSplat] Done. Preview 3DGS exports in pages/splat-viewer.html if format is .ply/.splat.")


if __name__ == "__main__":
    main()
