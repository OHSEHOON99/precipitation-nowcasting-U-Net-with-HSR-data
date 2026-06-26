import argparse
import gzip
import logging
import os
import tarfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm


def _is_within_directory(path, directory):
    path = Path(path).resolve()
    directory = Path(directory).resolve()
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _safe_extract_member(tar, member, output_dir):
    if member.issym() or member.islnk():
        raise ValueError(f"Refusing to extract linked archive member: {member.name}")
    target_path = Path(output_dir) / member.name
    if not _is_within_directory(target_path, output_dir):
        raise ValueError(f"Unsafe archive member path: {member.name}")
    tar.extract(member, output_dir)


def extract_archive(archive_path, output_dir):
    logging.info("Extracting %s", archive_path)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            _safe_extract_member(tar, member, output_dir)


def extract_archives(archives_dir, output_dir, n_jobs=-1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(Path(archives_dir).glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No .tar.gz files found in {archives_dir}")
    Parallel(n_jobs=n_jobs)(delayed(extract_archive)(archive, output_dir) for archive in archives)


def list_files(src_dir, suffix):
    return sorted(Path(src_dir).rglob(f"*{suffix}"))


def timestamp_from_hsr_path(path):
    stem = Path(path).name
    if stem.endswith(".bin.gz"):
        stem = stem[:-7]
    return stem[-12:]


def process_file(
        gz_path,
        output_dir,
        target_idx=(1439, 1214),
        size=256,
        raw_shape=(2881, 2305),
        header_bytes=1024,
):
    try:
        with gzip.open(gz_path, "rb") as file_obj:
            data = file_obj.read()

        row, col = target_idx
        dbz = np.frombuffer(data[header_bytes:], dtype=np.int16)
        dbz = dbz.reshape(raw_shape).copy()
        dbz[dbz < -1000] = -1000
        dbz = np.flipud(dbz + 1000)

        cropped = dbz[row:row + size, col:col + size]
        if cropped.shape != (size, size):
            raise ValueError(f"Crop shape {cropped.shape} does not match requested size {(size, size)}")

        timestamp = timestamp_from_hsr_path(gz_path)
        image_dir = Path(output_dir) / timestamp[:6] / timestamp[6:8]
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{timestamp}.png"
        Image.fromarray(cropped.astype(np.uint16)).save(image_path)
        return None
    except Exception as exc:
        logging.warning("Failed to process %s: %s", gz_path, exc)
        return str(gz_path)


def convert_bin_gz(raw_dir, output_dir, row=1439, col=1214, size=256, n_jobs=-1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(raw_dir, ".bin.gz")
    if not files:
        raise FileNotFoundError(f"No .bin.gz files found in {raw_dir}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file_path, output_dir, target_idx=(row, col), size=size)
        for file_path in tqdm(files, desc="Converting HSR binaries")
    )
    errors = [result for result in results if result is not None]
    if errors:
        error_log = output_dir / "conversion_errors.txt"
        error_log.write_text("\n".join(errors) + "\n", encoding="utf-8")
        logging.warning("Finished with %d conversion errors. See %s", len(errors), error_log)
    else:
        logging.info("Finished conversion without errors.")


def build_parser():
    parser = argparse.ArgumentParser(description="Extract KMA HSR archives and convert .bin.gz data to cropped PNG files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract .tar.gz archives")
    extract_parser.add_argument("--archives-dir", required=True, help="Directory containing .tar.gz archives")
    extract_parser.add_argument("--output-dir", required=True, help="Directory to receive extracted raw files")
    extract_parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for extraction")

    convert_parser = subparsers.add_parser("convert", help="Convert .bin.gz files to cropped 16-bit PNG files")
    convert_parser.add_argument("--raw-dir", required=True, help="Directory containing extracted .bin.gz files")
    convert_parser.add_argument("--output-dir", required=True, help="Directory to receive PNG files")
    convert_parser.add_argument("--row", type=int, default=1439, help="Top row of the crop")
    convert_parser.add_argument("--col", type=int, default=1214, help="Left column of the crop")
    convert_parser.add_argument("--size", type=int, default=256, help="Crop size in pixels")
    convert_parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs for conversion")

    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if args.command == "extract":
        extract_archives(args.archives_dir, args.output_dir, args.jobs)
    elif args.command == "convert":
        convert_bin_gz(args.raw_dir, args.output_dir, args.row, args.col, args.size, args.jobs)


if __name__ == "__main__":
    main()
