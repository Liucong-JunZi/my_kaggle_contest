#!/usr/bin/env python3
"""Convert .ipynb files in kernels_raw/ to code-only .py dumps for scanning."""
import json
import os
import sys

ROOT = "/Users/liucong/code/kaggle/ROGII/experiments/public_resources/kernels_raw"

def convert(path):
    out = path.replace(".ipynb", ".code.txt")
    if os.path.exists(out):
        return False
    try:
        with open(path) as f:
            nb = json.load(f)
    except Exception as e:
        print(f"FAIL {path}: {e}", file=sys.stderr)
        return False
    lines = []
    for cell in nb.get("cells", []):
        ct = cell.get("cell_type", "")
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        if ct == "markdown":
            lines.append("\n# === MARKDOWN CELL ===\n# " + src.replace("\n", "\n# ") + "\n")
        elif ct == "code":
            lines.append("\n# === CODE CELL ===\n" + src + "\n")
    with open(out, "w") as f:
        f.write("".join(lines))
    return True

def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    converted = 0
    for d in sorted(os.listdir(ROOT)):
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        if only and d not in only:
            continue
        for fn in os.listdir(full):
            if fn.endswith(".ipynb"):
                if convert(os.path.join(full, fn)):
                    converted += 1
    print(f"converted {converted} notebooks")

if __name__ == "__main__":
    main()
