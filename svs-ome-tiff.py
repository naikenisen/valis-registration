#!/usr/bin/env python3
"""Convert .svs whole-slide images to .ome.tiff.

nohup python -u svs-ome-tiff.py --input-dir ~/silver/ube/slides --output-dir ~/silver/ube/slides_ome_tiff > conversion.log 2>&1 &
"""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import sys
from collections import defaultdict


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Batch convert SVS files to OME-TIFF using pyvips."
	)
	parser.add_argument(
		"--input-dir",
		default="~/silver/ube/slides",
		help="Directory containing .svs files (default: ~/silver/ube/slides)",
	)
	parser.add_argument(
		"--output-dir",
		default="~/silver/ube/slides_ome_tiff",
		help="Directory where .ome.tiff files are written (default: ~/silver/ube/slides_ome_tiff)",
	)
	parser.add_argument(
		"--compression",
		default="jpeg",
		choices=["jpeg", "lzw", "deflate", "zstd", "none"],
		help="TIFF compression method (default: jpeg)",
	)
	parser.add_argument(
		"--quality",
		type=int,
		default=90,
		help="JPEG quality if compression=jpeg (default: 90)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Re-convert files even if the .ome.tiff output already exists.",
	)
	return parser.parse_args()


def convert_file(
	svs_path: str,
	output_dir: str,
	compression: str,
	quality: int,
	overwrite: bool,
) -> tuple[str, str]:
	try:
		pyvips = importlib.import_module("pyvips")
	except ImportError:
		return "error", (
			"Missing dependency: pyvips. Install it in your venv with `pip install pyvips`."
		)

	base_name = os.path.splitext(os.path.basename(svs_path))[0]
	out_path = os.path.join(output_dir, f"{base_name}.ome.tiff")

	if os.path.exists(out_path) and not overwrite:
		return "skip", out_path

	try:
		# Opens the first (full-resolution) page of the SVS image.
		image = pyvips.Image.new_from_file(svs_path, page=0, access="sequential")

		kwargs = {
			"tile": True,
			"tile_width": 512,
			"tile_height": 512,
			"pyramid": True,
			"bigtiff": True,
			"subifd": True,
		}

		if compression != "none":
			kwargs["compression"] = compression
		if compression == "jpeg":
			kwargs["Q"] = quality

		image.tiffsave(out_path, **kwargs)
		return "ok", out_path
	except Exception as exc:  # pylint: disable=broad-except
		return "error", f"{svs_path}: {exc}"


def remove_orphan_outputs(output_dir: str) -> int:
	"""Remove output files that do not have a paired file.

	Pairing is inferred from the prefix before the first underscore in filename,
	e.g. `a_CD30.ome.tiff` pairs with `a_HES.ome.tiff`.
	"""
	output_files = sorted(glob.glob(os.path.join(output_dir, "*.ome.tiff")))
	grouped: dict[str, list[str]] = defaultdict(list)

	for out_path in output_files:
		base_name = os.path.basename(out_path)
		root_name = base_name.removesuffix(".ome.tiff")
		pair_key = root_name.split("_", maxsplit=1)[0]
		grouped[pair_key].append(out_path)

	removed = 0
	for pair_key, files in sorted(grouped.items()):
		if len(files) == 1:
			orphan_path = files[0]
			os.remove(orphan_path)
			removed += 1
			print(f"  CLEANUP -> removed orphan (missing pair '{pair_key}_*'): {orphan_path}")

	return removed


def main() -> int:
	# Ensure progress logs appear immediately when stdout/stderr are redirected.
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(line_buffering=True)
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(line_buffering=True)

	args = parse_args()

	input_dir = os.path.expanduser(args.input_dir)
	output_dir = os.path.expanduser(args.output_dir)

	os.makedirs(output_dir, exist_ok=True)
	print("Checking existing outputs before resume...")
	removed_orphans = remove_orphan_outputs(output_dir)
	if removed_orphans:
		print(f"Pre-check complete. Removed orphan output file(s): {removed_orphans}.")
	else:
		print("Pre-check complete. No orphan outputs found.")

	svs_files = sorted(glob.glob(os.path.join(input_dir, "*.svs")))
	if not svs_files:
		print(f"No .svs files found in: {input_dir}")
		return 1

	print(f"Found {len(svs_files)} SVS file(s) in {input_dir}")
	print(f"Output directory: {output_dir}")

	success_count = 0
	skipped_count = 0
	error_count = 0
	for idx, svs_path in enumerate(svs_files, start=1):
		print(f"[{idx}/{len(svs_files)}] Converting {os.path.basename(svs_path)} ...")
		status, message = convert_file(
			svs_path=svs_path,
			output_dir=output_dir,
			compression=args.compression,
			quality=args.quality,
			overwrite=args.overwrite,
		)
		if status == "ok":
			success_count += 1
			print(f"  OK -> {message}")
		elif status == "skip":
			skipped_count += 1
			print(f"  SKIP -> already exists: {message}")
		else:
			error_count += 1
			print(f"  ERROR -> {message}")

	print(
		"Done. "
		f"Converted: {success_count}, "
		f"Skipped(existing): {skipped_count}, "
		f"Errors: {error_count}."
	)

	return 0 if error_count == 0 else 2


if __name__ == "__main__":
	sys.exit(main())
