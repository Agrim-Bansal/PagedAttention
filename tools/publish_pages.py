#!/usr/bin/env python3
"""Publish dist/ to the gh-pages branch without switching the working tree.

Copies the HTML export as index.html (plus the original filename), the video
assets, and .nojekyll, then writes a commit on gh-pages and optionally pushes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
HTML = DIST / "pagedattention_talk.html"
ASSETS = DIST / "pagedattention_talk_assets"
BRANCH = "gh-pages"
COMMIT_MESSAGE = "Publish HTML talk to GitHub Pages"


def run(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        capture_output=True,
    )


def fail_if_missing() -> None:
    if not HTML.is_file():
        sys.exit(f"Missing {HTML.relative_to(ROOT)}. Run: make html")
    if not ASSETS.is_dir():
        sys.exit(f"Missing {ASSETS.relative_to(ROOT)}. Run: make html")


def stage_site(staging: Path) -> None:
    shutil.copy2(HTML, staging / "index.html")
    shutil.copy2(HTML, staging / HTML.name)
    shutil.copytree(ASSETS, staging / ASSETS.name)
    (staging / ".nojekyll").write_text("")
    (staging / "README.md").write_text(
        "# PagedAttention talk\n\n"
        "Generated GitHub Pages site. Do not edit; republish with `make pages`.\n"
    )


def current_branch_sha() -> str | None:
    result = run(
        ["git", "rev-parse", "--verify", f"refs/heads/{BRANCH}"],
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def tree_for(commit: str) -> str:
    return run(["git", "rev-parse", f"{commit}^{{tree}}"]).stdout.strip()


def publish(*, push: bool) -> None:
    fail_if_missing()

    staging = Path(tempfile.mkdtemp(prefix="pagedattention-pages-"))
    index_path: Path | None = None
    try:
        stage_site(staging)

        index_path = Path(tempfile.mkdtemp(prefix="gh-pages-index-")) / "index"

        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index_path)
        env["GIT_WORK_TREE"] = str(staging)

        run(["git", "add", "-A"], env=env)
        tree = run(["git", "write-tree"], env=env).stdout.strip()

        parent = current_branch_sha()
        if parent is not None and tree_for(parent) == tree:
            print(f"{BRANCH} already up to date ({parent[:12]})")
            if push:
                run(["git", "push", "-u", "origin", BRANCH])
                print(f"Pushed {BRANCH} to origin")
            return

        commit_cmd = ["git", "commit-tree", tree, "-m", COMMIT_MESSAGE]
        if parent is not None:
            commit_cmd.extend(["-p", parent])
        commit = run(commit_cmd).stdout.strip()
        run(["git", "update-ref", f"refs/heads/{BRANCH}", commit])
        print(f"Updated {BRANCH} -> {commit}")

        if push:
            run(["git", "push", "-u", "origin", BRANCH])
            print(f"Pushed {BRANCH} to origin")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        sys.exit(detail or str(exc))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if index_path is not None:
            shutil.rmtree(index_path.parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Update the local gh-pages branch only",
    )
    args = parser.parse_args()
    publish(push=not args.no_push)


if __name__ == "__main__":
    main()
