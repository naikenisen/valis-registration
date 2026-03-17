"""registration.py
nohup python registration.py --input-dir /silver/ube/slides_ome_tiff \
    --output-dir /silver/ube/registration_results
     > conversion.log 2>&1 &
"""


from valis import registration
import os
import glob

slide_src_dir = os.path.expanduser("~/coding/silver/slides")
base_results_dst_dir = os.path.expanduser("~/coding/silver/registration_results")

# 1. Lister toutes les lames HES pour identifier les préfixes (a, b, c...)
hes_slides = glob.glob(os.path.join(slide_src_dir, "*_HES.svs"))

for hes_path in hes_slides:
    filename = os.path.basename(hes_path)
    prefix = filename.split('_')[0] 
    
    cd30_path = os.path.join(slide_src_dir, f"{prefix}_CD30.svs")

    # SÉCURITÉ : Vérifier que le fichier CD30 existe bien avant de lancer VALIS
    if not os.path.exists(cd30_path):
        print(f"⚠️ Fichier manquant pour {prefix}: {cd30_path} n'existe pas. Ignoré.")
        continue

    print(f"\n--- Alignement de la paire : {prefix} ---")
    
    pair_results_dir = os.path.join(base_results_dst_dir, prefix)
    img_list = [hes_path, cd30_path]
    
    registrar = registration.Valis(
        src_dir=slide_src_dir, 
        dst_dir=pair_results_dir, 
        img_list=img_list,
        reference_img_f=hes_path
    )
    
    # Lancer l'enregistrement (calcule les transformations)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    
    # Sauvegarde des images complètes superposables (format .ome.tiff)
    registered_slide_dst_dir = os.path.join(pair_results_dir, "registered_slides")
    registrar.warp_and_save_slides(registered_slide_dst_dir)
    
    print(f"✅ Paire {prefix} terminée avec succès.")

print("\n🚀 Traitement total terminé !")