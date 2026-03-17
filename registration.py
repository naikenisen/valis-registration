"""registration.py
nohup python registration.py > registration.log 2>&1 &
"""

import os
# Force CPU to avoid VALIS/LightGlue CUDA tensor->NumPy conversion errors.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from valis import registration

patient = "a"
hes_path = f"/silver/ube/slides_ome_tiff/{patient}_HES.ome.tiff"
cd30_path = f"/silver/ube/slides_ome_tiff/{patient}_CD30.ome.tiff"
base_results_dst_dir = "/silver/ube/registration_results_v2"

os.makedirs(base_results_dst_dir, exist_ok=True)


pair_results_dir = os.path.join(base_results_dst_dir, patient)
os.makedirs(pair_results_dir, exist_ok=True)

# Use parent folder of the input slides as VALIS source directory.
img_list = [hes_path, cd30_path]

print(f"Aligning pair")

registrar = registration.Valis(
    src_dir="/silver/ube/slides_ome_tiff/",
    dst_dir=pair_results_dir,
    img_list=img_list,
    reference_img_f=hes_path,
    max_processed_image_dim_px = 1600,
)

rigid_registrar, non_rigid_registrar, error_df = registrar.register()

registered_slide_dst_dir = os.path.join(pair_results_dir, "registered_slides")
registrar.warp_and_save_slides(registered_slide_dst_dir)

print(f"Pair completed successfully.")
registration.kill_jvm()
