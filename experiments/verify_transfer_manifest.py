"""Verify every file listed in a transfer manifest without modifying the package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "TRANSFER_MANIFEST.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for item in manifest["files"]:
        path = root / item["relative_path"]
        if not path.is_file():
            failures.append(f"missing: {item['relative_path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"size mismatch: {item['relative_path']}")
            continue
        if sha256(path) != item["sha256"]:
            failures.append(f"hash mismatch: {item['relative_path']}")
    report = {
        "overall_pass": not failures,
        "files_checked": len(manifest["files"]),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
