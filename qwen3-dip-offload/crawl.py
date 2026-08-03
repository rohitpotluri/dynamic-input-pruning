#!/usr/bin/env python3
"""
crawl.py — Dump the directory structure and the contents of every text-based
file under the current directory into a single output file (crawl_output.txt).

Usage:
    python crawl.py [root_dir] [-o output.txt]

Defaults: root_dir = "." (the folder crawl.py lives in), output = crawl_output.txt
"""

import os
import argparse

# Files (by exact name) to skip in the interest of length.
SKIP_NAMES = {
    "LICENSE",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
}

# Directories to skip entirely.
SKIP_DIRS = {
    ".git", "__pycache__", ".idea", ".vscode", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "venv",
    "checkpoints",  # often huge model dirs in gpt-fast; remove if you want them
}

# Extensions we treat as text and will dump contents for.
TEXT_EXTS = {
    ".py", ".sh", ".md", ".txt", ".cfg", ".ini", ".toml", ".yaml", ".yml",
    ".json", ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".rs", ".js", ".ts",
    ".html", ".css", ".env", ".gitignore", ".bat", ".make", "",  # "" = extensionless
}

# Files larger than this (bytes) get their content skipped (path still listed).
MAX_BYTES = 1_000_000


def is_text_file(name: str) -> bool:
    _, ext = os.path.splitext(name)
    if name in {"Makefile", "Dockerfile", "requirements.txt"}:
        return True
    return ext.lower() in TEXT_EXTS


def build_tree(root: str) -> str:
    lines = []
    root = os.path.abspath(root)
    root_name = os.path.basename(root) or root
    lines.append(root_name + "/")

    def walk(dir_path: str, prefix: str):
        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            return
        entries = [e for e in entries if e not in SKIP_DIRS]
        dirs = [e for e in entries if os.path.isdir(os.path.join(dir_path, e))]
        files = [e for e in entries if not os.path.isdir(os.path.join(dir_path, e))]
        ordered = dirs + files
        for i, entry in enumerate(ordered):
            full = os.path.join(dir_path, entry)
            is_last = (i == len(ordered) - 1)
            connector = "└── " if is_last else "├── "
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(prefix + connector + entry + suffix)
            if os.path.isdir(full):
                extension = "    " if is_last else "│   "
                walk(full, prefix + extension)

    walk(root, "")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", help="Directory to crawl")
    parser.add_argument("-o", "--output", default="crawl_output.txt",
                        help="Output file name")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    out_path = os.path.abspath(args.output)

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("DIRECTORY STRUCTURE\n\n")
        out.write(build_tree(root) + "\n\n")

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in sorted(filenames):
                if fname in SKIP_NAMES:
                    continue
                full = os.path.join(dirpath, fname)
                # Don't dump the output file or the crawler itself.
                if os.path.abspath(full) in {out_path}:
                    continue
                rel = os.path.relpath(full, root)

                if not is_text_file(fname):
                    out.write(f"\n[skipped non-text: {rel}]\n")
                    continue

                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue

                out.write(f"\n=== {rel} ===\n")

                if size > MAX_BYTES:
                    out.write(f"[skipped: {size} bytes > {MAX_BYTES} limit]\n")
                    continue

                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"[error reading file: {e}]\n")
                out.write("\n")

    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()