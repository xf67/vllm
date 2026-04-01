#!/usr/bin/env python3
"""Merge one experiment result folder into another.

Usage:
    python merge_bench_results.py SRC_DIR DST_DIR [--output-dir OUTPUT_DIR]

Example:
    python merge_bench_results.py \
        vllm/test-vllm-ok2/bench_results \
        vllm/test-vllm-ok/bench_results \
        --output-dir vllm/test-vllm-merged/bench_results

Behavior:
    - By default, recursively copies everything from SRC_DIR into DST_DIR.
    - If --output-dir is provided, the script first copies DST_DIR to
      OUTPUT_DIR, then overlays SRC_DIR onto OUTPUT_DIR.
    - If a file already exists in the merge target, it is overwritten by SRC_DIR.
    - Extra files that exist only in DST_DIR are kept.
    - If a path type conflicts (file vs directory), the target path is
      replaced so the source tree can be merged cleanly.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return

    print(f"[REPLACE] remove conflicting destination: {path}")
    if dry_run:
        return

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def ensure_directory(path: Path, dry_run: bool, created_dirs: list[int]) -> None:
    if path.exists():
        return

    print(f"[MKDIR]   {path}")
    created_dirs[0] += 1
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def copy_tree(src_root: Path, dst_root: Path, dry_run: bool) -> tuple[int, int]:
    created_dirs = [0]
    copied_files = 0

    ensure_directory(dst_root, dry_run=dry_run, created_dirs=created_dirs)

    for current_root, dirnames, filenames in os.walk(src_root):
        current_root = Path(current_root)
        rel_root = current_root.relative_to(src_root)
        dst_dir = dst_root / rel_root
        ensure_directory(dst_dir, dry_run=dry_run, created_dirs=created_dirs)

        for dirname in dirnames:
            src_dir = current_root / dirname
            rel_dir = src_dir.relative_to(src_root)
            dst_subdir = dst_root / rel_dir
            ensure_directory(dst_subdir, dry_run=dry_run, created_dirs=created_dirs)

        for filename in filenames:
            src_file = current_root / filename
            rel_file = src_file.relative_to(src_root)
            dst_file = dst_root / rel_file

            print(f"[BASE COPY] {src_file} -> {dst_file}")
            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
            copied_files += 1

    return created_dirs[0], copied_files


def merge_trees(src_root: Path, dst_root: Path, dry_run: bool) -> tuple[int, int, int]:
    created_dirs = [0]
    copied_files = 0
    overwritten_files = 0

    ensure_directory(dst_root, dry_run=dry_run, created_dirs=created_dirs)

    for current_root, dirnames, filenames in os.walk(src_root):
        current_root = Path(current_root)
        rel_root = current_root.relative_to(src_root)
        dst_dir = dst_root / rel_root

        if dst_dir.exists() and not dst_dir.is_dir():
            remove_path(dst_dir, dry_run=dry_run)

        ensure_directory(dst_dir, dry_run=dry_run, created_dirs=created_dirs)

        for dirname in dirnames:
            src_dir = current_root / dirname
            rel_dir = src_dir.relative_to(src_root)
            dst_subdir = dst_root / rel_dir

            if dst_subdir.exists() and not dst_subdir.is_dir():
                remove_path(dst_subdir, dry_run=dry_run)

            ensure_directory(dst_subdir, dry_run=dry_run, created_dirs=created_dirs)

        for filename in filenames:
            src_file = current_root / filename
            rel_file = src_file.relative_to(src_root)
            dst_file = dst_root / rel_file

            if dst_file.exists() and dst_file.is_dir():
                remove_path(dst_file, dry_run=dry_run)

            action = "OVERWRITE" if dst_file.exists() else "COPY"
            print(f"[{action}] {src_file} -> {dst_file}")

            if not dry_run:
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)

            if action == "OVERWRITE":
                overwritten_files += 1
            else:
                copied_files += 1

    return created_dirs[0], copied_files, overwritten_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SRC_DIR into DST_DIR by overwriting destination files.",
    )
    parser.add_argument(
        "src_dir",
        type=Path,
        help="Source result folder. Its files overwrite the destination.",
    )
    parser.add_argument(
        "dst_dir",
        type=Path,
        help="Base result folder. SRC_DIR overwrites this folder or its copied output.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Write merged results to a new folder instead of modifying DST_DIR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without modifying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = args.src_dir.expanduser().resolve()
    dst_dir = args.dst_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else None
    )

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_dir}")
    if not dst_dir.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {dst_dir}")
    if not dst_dir.is_dir():
        raise NotADirectoryError(f"Destination path is not a directory: {dst_dir}")
    if src_dir == dst_dir:
        raise ValueError("Source and destination directories must be different.")

    # Prevent copying into a nested destination, which would recurse forever.
    if is_relative_to(dst_dir, src_dir):
        raise ValueError(
            "Destination directory cannot be inside source directory.",
        )

    base_copied_dirs = 0
    base_copied_files = 0
    merge_target = dst_dir

    if output_dir is not None:
        if output_dir.exists():
            raise FileExistsError(
                f"Output directory already exists, please use a new path: {output_dir}",
            )
        if is_relative_to(output_dir, src_dir):
            raise ValueError("Output directory cannot be inside source directory.")
        if is_relative_to(output_dir, dst_dir):
            raise ValueError("Output directory cannot be inside destination directory.")

        merge_target = output_dir
        base_copied_dirs, base_copied_files = copy_tree(
            src_root=dst_dir,
            dst_root=output_dir,
            dry_run=args.dry_run,
        )

    created_dirs, copied_files, overwritten_files = merge_trees(
        src_root=src_dir,
        dst_root=merge_target,
        dry_run=args.dry_run,
    )

    print("\nDone.")
    print(f"  Source:      {src_dir}")
    print(f"  Base:        {dst_dir}")
    print(f"  Output:      {merge_target}")
    if output_dir is not None:
        print(f"  Base dirs copied:   {base_copied_dirs}")
        print(f"  Base files copied:  {base_copied_files}")
    print(f"  Directories created: {created_dirs}")
    print(f"  Files copied:        {copied_files}")
    print(f"  Files overwritten:   {overwritten_files}")
    if args.dry_run:
        print("  Mode: dry-run only, no files were changed.")


if __name__ == "__main__":
    main()


# python3 vllm/experiment/merge_bench_results.py vllm/test-vllm-ok2/bench_results vllm/test-vllm-ok/bench_results --output-dir vllm/test-vllm-merged/bench_results