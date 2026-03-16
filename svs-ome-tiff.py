#!/usr/bin/env python3
"""Convert .svs whole-slide images to .ome.tiff.

Usage example:
	python svs-ome-tiff.py \
		--input-dir ~/coding/silver/slides \
		--output-dir ~/coding/silver/slides_ome_tiff
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Batch convert SVS files to OME-TIFF using pyvips."
	)
	parser.add_argument(
		"--input-dir",
		default="~/coding/silver/slides",
		help="Directory containing .svs files (default: ~/coding/silver/slides)",
	)
	parser.add_argument(
		"--output-dir",
		default="~/coding/silver/ome_tiff",
		help="Directory where .ome.tiff files are written (default: ~/coding/silver/ome_tiff)",
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
	return parser.parse_args()


def convert_file(
	svs_path: str,
	output_dir: str,
	compression: str,
	quality: int,
) -> tuple[bool, str]:
	try:
		import pyvips
	except ImportError:
		return False, (
			"Missing dependency: pyvips. Install it in your venv with `pip install pyvips`."
		)

	base_name = os.path.splitext(os.path.basename(svs_path))[0]
	out_path = os.path.join(output_dir, f"{base_name}.ome.tiff")

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
		return True, out_path
	except Exception as exc:  # pylint: disable=broad-except
		return False, f"{svs_path}: {exc}"


def main() -> int:
	args = parse_args()

	input_dir = os.path.expanduser(args.input_dir)
	output_dir = os.path.expanduser(args.output_dir)

	os.makedirs(output_dir, exist_ok=True)

	svs_files = sorted(glob.glob(os.path.join(input_dir, "*.svs")))
	if not svs_files:
		print(f"No .svs files found in: {input_dir}")
		return 1

	print(f"Found {len(svs_files)} SVS file(s) in {input_dir}")
	print(f"Output directory: {output_dir}")

	success_count = 0
	for idx, svs_path in enumerate(svs_files, start=1):
		print(f"[{idx}/{len(svs_files)}] Converting {os.path.basename(svs_path)} ...")
		ok, message = convert_file(
			svs_path=svs_path,
			output_dir=output_dir,
			compression=args.compression,
			quality=args.quality,
		)
		if ok:
			success_count += 1
			print(f"  OK -> {message}")
		else:
			print(f"  ERROR -> {message}")

	print(f"Done. Converted {success_count}/{len(svs_files)} file(s).")
	return 0 if success_count == len(svs_files) else 2


if __name__ == "__main__":
	sys.exit(main())
