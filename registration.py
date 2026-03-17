"""registration.py
nohup python registration.py > registration.log 2>&1 &
"""

import os

# Force CPU to avoid VALIS/LightGlue CUDA tensor->NumPy conversion errors.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from valis import registration

# Hardcoded pair to register
HES_SLIDE_PATH = "/silver/ube/slides_ome_tiff/a_HES.ome.tiff"
CD30_SLIDE_PATH = "/silver/ube/slides_ome_tiff/a_CD30.ome.tiff"
OUTPUT_DIR = "/silver/ube/registration_results_v2"


def main():
    hes_path = os.path.expanduser(HES_SLIDE_PATH)
    cd30_path = os.path.expanduser(CD30_SLIDE_PATH)
    base_results_dst_dir = os.path.expanduser(OUTPUT_DIR)

    if not os.path.isfile(hes_path):
        raise FileNotFoundError(f"HES slide not found: {hes_path}")

    if not os.path.isfile(cd30_path):
        raise FileNotFoundError(f"CD30 slide not found: {cd30_path}")

    os.makedirs(base_results_dst_dir, exist_ok=True)

    filename = os.path.basename(hes_path)
    prefix = filename.rsplit("_HES", 1)[0]
    pair_results_dir = os.path.join(base_results_dst_dir, prefix)

    # Use parent folder of the input slides as VALIS source directory.
    slide_src_dir = os.path.dirname(hes_path)
    img_list = [hes_path, cd30_path]

    try:
        print(f"\n--- Aligning pair: {prefix} ---")

        registrar = registration.Valis(
            src_dir=slide_src_dir,
            dst_dir=pair_results_dir,
            img_list=img_list,
            reference_img_f=hes_path,
        )

        rigid_registrar, _, _ = registrar.register()

        if rigid_registrar is None:
            print(f"Registration failed for {prefix}.")
            return

        registered_slide_dst_dir = os.path.join(pair_results_dir, "registered_slides")
        registrar.warp_and_save_slides(registered_slide_dst_dir)

        print(f"Pair {prefix} completed successfully.")
    finally:
        # Ensure Bio-Formats JVM is terminated cleanly.
        registration.kill_jvm()


if __name__ == "__main__":
    main()