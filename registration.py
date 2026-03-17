"""registration.py
nohup python registration.py \
    --input-dir /silver/ube/slides_ome_tiff \
    --output-dir /silver/ube/registration_results \
    > registration.log 2>&1 &
"""

import argparse
import glob
import os

from valis import registration

SUPPORTED_EXTENSION = ".ome.tiff"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register HES/CD30 slide pairs with VALIS"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing slides named like <prefix>_HES and <prefix>_CD30",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where registration results are written",
    )
    return parser.parse_args()


def find_hes_slides(slide_src_dir):
    return sorted(
        glob.glob(os.path.join(slide_src_dir, f"*_HES{SUPPORTED_EXTENSION}"))
    )


def find_cd30_slide(slide_src_dir, prefix):
    candidate = os.path.join(slide_src_dir, f"{prefix}_CD30{SUPPORTED_EXTENSION}")
    return candidate if os.path.exists(candidate) else None


def main():
    args = parse_args()

    slide_src_dir = os.path.expanduser(args.input_dir)
    base_results_dst_dir = os.path.expanduser(args.output_dir)

    if not os.path.isdir(slide_src_dir):
        raise FileNotFoundError(f"Input directory not found: {slide_src_dir}")

    os.makedirs(base_results_dst_dir, exist_ok=True)

    hes_slides = find_hes_slides(slide_src_dir)
    if not hes_slides:
        print(
            f"No HES slides found in {slide_src_dir} for extension: {SUPPORTED_EXTENSION}"
        )
        return

    for hes_path in hes_slides:
        filename = os.path.basename(hes_path)
        prefix = filename.rsplit("_HES", 1)[0]

        cd30_path = find_cd30_slide(slide_src_dir, prefix)

        if cd30_path is None:
            print(f"Missing CD30 slide for {prefix}. Skipped.")
            continue

        print(f"\n--- Aligning pair: {prefix} ---")

        pair_results_dir = os.path.join(base_results_dst_dir, prefix)
        img_list = [hes_path, cd30_path]

        registrar = registration.Valis(
            src_dir=slide_src_dir,
            dst_dir=pair_results_dir,
            img_list=img_list,
            reference_img_f=hes_path,
        )

        registrar.register()

        registered_slide_dst_dir = os.path.join(pair_results_dir, "registered_slides")
        registrar.warp_and_save_slides(registered_slide_dst_dir)

        print(f"Pair {prefix} completed successfully.")

    print("\nTotal processing completed.")


if __name__ == "__main__":
    main()