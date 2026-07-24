"""Build a clean SHA-256 manifest for the deadline-adapted workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    json_output = args.json_output.resolve()
    csv_output = args.csv_output.resolve()
    output_paths = {json_output, csv_output}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in output_paths:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(
            {
                "relative_path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "root": str(root),
        "exclusions": {
            "path_parts": sorted(EXCLUDED_PARTS),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("relative_path", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(files)
    print(
        json.dumps(
            {
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "json": str(json_output),
                "csv": str(csv_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
