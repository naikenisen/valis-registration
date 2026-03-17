"""registration.py
nohup python registration.py > registration.log 2>&1 &
"""

import os
# Force CPU to avoid VALIS/LightGlue CUDA tensor->NumPy conversion errors.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
from valis import registration, affine_optimizer
from valis.micro_rigid_registrar import MicroRigidRegistrar

# gestion des chemins
patient = "a"
hes_path = f"/silver/ube/slides_ome_tiff/{patient}_HES.ome.tiff"
cd30_path = f"/silver/ube/slides_ome_tiff/{patient}_CD30.ome.tiff"
base_results_dst_dir = "/silver/ube/registration_results_v2"
pair_results_dir = os.path.join(base_results_dst_dir, patient)
os.makedirs(pair_results_dir, exist_ok=True)
registered_slide_dst_dir = os.path.join(pair_results_dir, "registered_slides")

img_list = [hes_path, cd30_path]

registrar = registration.Valis(
    src_dir="/silver/ube/slides_ome_tiff/",
    dst_dir=pair_results_dir,
    img_list=img_list,
    reference_img_f=hes_path,
    max_processed_image_dim_px=2000,
    align_to_reference=True,
    max_non_rigid_registration_dim_px=2048,
    micro_rigid_registrar_cls=MicroRigidRegistrar,
    affine_optimizer_cls=affine_optimizer.AffineOptimizerMattesMI,
    denoise_rigid=True,
)

print(f"Aligning pair")

rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Micro non-rigid registration at 25% of full resolution
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
min_max_size = np.min([np.max(d) for d in img_dims])
micro_reg_size = int(np.floor(min_max_size * 0.25))
print(f"Running micro non-rigid registration at {micro_reg_size}px")
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)

# registrar.warp_and_save_slides(registered_slide_dst_dir)

print(f"Pair completed successfully.")
registration.kill_jvm()
