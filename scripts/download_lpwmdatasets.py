#!/usr/bin/env python3
"""Download the exact DINO-WM/LpWM PushT and Wall datasets from OSF.

Archives are streamed to local storage, resumed when possible, verified against
the SHA-256 values published by OSF, and extracted with path-traversal checks.
The training code still consumes the original directory layout through
``DATASET_DIR``; this script does not transform the data.
"""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm


OSF_VIEW_ONLY = "a56a296ce3b24cceaf408383a175ce28"
DATASETS = {
    "pusht_noise": {
        "url": f"https://osf.io/download/k2d8w/?view_only={OSF_VIEW_ONLY}",
        "size": 2_785_304_515,
        "sha256": "442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08",
        "markers": (
            "train/states.pth",
            "train/rel_actions.pth",
            "train/seq_lengths.pkl",
            "train/velocities.pth",
            "train/obses",
            "val/states.pth",
            "val/rel_actions.pth",
            "val/seq_lengths.pkl",
            "val/velocities.pth",
            "val/obses",
        ),
    },
    "wall_single": {
        "url": f"https://osf.io/download/49rnx/?view_only={OSF_VIEW_ONLY}",
        "size": 1_668_205_895,
        "sha256": "2b4ae4ed0ad03b337efac637f17752e7e7e27f864fec39dc25b51fef490c980d",
        "markers": (
            "states.pth",
            "actions.pth",
            "door_locations.pth",
            "wall_locations.pth",
            "obses",
        ),
    },
}


def human_size(size):
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_markers(directory, markers):
    directory = Path(directory)
    return [marker for marker in markers if not (directory / marker).exists()]


def dataset_is_ready(output_dir, name):
    spec = DATASETS[name]
    return not missing_markers(Path(output_dir) / name, spec["markers"])


def download_archive(name, archive_path, force=False):
    spec = DATASETS[name]
    archive_path = Path(archive_path)
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        valid_size = archive_path.stat().st_size == spec["size"]
        if valid_size and sha256_file(archive_path) == spec["sha256"]:
            print(f"Using verified archive: {archive_path}")
            return archive_path
        if not force:
            raise RuntimeError(
                f"Existing archive is invalid: {archive_path}. Re-run with --force to replace it."
            )
        archive_path.unlink()

    if force and partial_path.exists():
        partial_path.unlink()

    offset = partial_path.stat().st_size if partial_path.exists() else 0
    if offset > spec["size"]:
        raise RuntimeError(f"Partial archive is larger than expected: {partial_path}")

    if offset < spec["size"]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = requests.get(spec["url"], headers=headers, stream=True, timeout=(30, 300))
        if offset and response.status_code != 206:
            response.close()
            partial_path.unlink()
            offset = 0
            response = requests.get(spec["url"], stream=True, timeout=(30, 300))
        response.raise_for_status()

        mode = "ab" if offset else "wb"
        try:
            with partial_path.open(mode) as handle, tqdm(
                total=spec["size"],
                initial=offset,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {name}",
            ) as progress:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        finally:
            response.close()

    actual_size = partial_path.stat().st_size
    if actual_size != spec["size"]:
        raise RuntimeError(
            f"Incomplete {name} archive: got {actual_size}, expected {spec['size']} bytes. "
            "Run the same command again to resume."
        )
    print(f"Verifying SHA-256 for {name} ...")
    actual_hash = sha256_file(partial_path)
    if actual_hash != spec["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {partial_path}: {actual_hash}")
    os.replace(partial_path, archive_path)
    return archive_path


def safe_extract_zip(archive_path, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if os.path.commonpath((str(destination), str(target))) != str(destination):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
            file_mode = (member.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(file_mode):
                raise RuntimeError(f"Symbolic link is not allowed in archive: {member.filename}")
        for member in tqdm(members, desc="Extracting", unit="file"):
            archive.extract(member, destination)


def find_extracted_dataset(extract_dir, name):
    spec = DATASETS[name]
    extract_dir = Path(extract_dir)
    candidates = [extract_dir / name, extract_dir]
    candidates.extend(path for path in extract_dir.rglob(name) if path.is_dir())
    for candidate in candidates:
        if candidate.is_dir() and not missing_markers(candidate, spec["markers"]):
            return candidate
    raise RuntimeError(
        f"Archive extracted, but the expected {name} layout was not found under {extract_dir}."
    )


def install_dataset(name, output_dir, keep_archive=False, force=False):
    output_dir = Path(output_dir).expanduser().resolve()
    target = output_dir / name
    spec = DATASETS[name]
    if dataset_is_ready(output_dir, name):
        print(f"Dataset already ready: {target}")
        return target
    if target.exists():
        missing = ", ".join(missing_markers(target, spec["markers"]))
        raise RuntimeError(
            f"Incomplete dataset directory exists at {target}; missing: {missing}. "
            "Move it aside or remove it explicitly before retrying."
        )

    archive_dir = output_dir / ".archives"
    archive_path = archive_dir / f"{name}.zip"
    archive_path = download_archive(name, archive_path, force=force)
    extract_dir = output_dir / f".extracting_{name}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    try:
        safe_extract_zip(archive_path, extract_dir)
        source = find_extracted_dataset(extract_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        metadata = {
            "dataset": name,
            "source": spec["url"],
            "bytes": spec["size"],
            "sha256": spec["sha256"],
        }
        (target / ".lpwm_source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
    if not keep_archive:
        archive_path.unlink(missing_ok=True)
    print(f"Ready: {target}")
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS.keys(), "all"),
        default="pusht_noise",
        help="Download one dataset or both LpWM datasets (default: pusht_noise).",
    )
    parser.add_argument("--output-dir", default="/content/lpwm-data")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace an invalid archive.")
    parser.add_argument("--list", action="store_true", help="List sources and exit.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, spec in DATASETS.items():
            print(f"{name}: {human_size(spec['size'])}  {spec['url']}")
        return 0
    names = DATASETS.keys() if args.dataset == "all" else (args.dataset,)
    for name in names:
        install_dataset(name, args.output_dir, args.keep_archive, args.force)
    print(f"Set DATASET_DIR={Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
