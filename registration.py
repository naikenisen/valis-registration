from valis import registration
import os
import glob

slide_src_dir = os.path.expanduser("~/coding/silver/slides")
base_results_dst_dir = "./slide_registration_example"

# 1. Lister toutes les lames HES pour identifier les préfixes (a, b, c...)
hes_slides = glob.glob(os.path.join(slide_src_dir, "*_HES.svs"))

for hes_path in hes_slides:
    # Extraire le préfixe (ex: "a" depuis "a_HES.svs")
    filename = os.path.basename(hes_path)
    prefix = filename.split('_')[0] 
    
    # Construire le chemin du fichier CD30 attendu pour cette paire
    cd30_path = os.path.join(slide_src_dir, f"{prefix}_CD30.svs")
    
    # Vérifier que le fichier CD30 existe bien
    if not os.path.exists(cd30_path):
        print(f"⚠️ Attention: {cd30_path} introuvable pour {filename}. On passe.")
        continue
        
    print(f"\n--- Alignement de la paire : {prefix} ---")
    
    # 2. Créer un sous-dossier de résultats spécifique pour cette paire
    pair_results_dir = os.path.join(base_results_dst_dir, prefix)
    
    # 3. Restreindre VALIS à cette liste exacte de 2 images
    img_list = [hes_path, cd30_path]
    
    # 4. Initialiser VALIS pour cette paire spécifique
    registrar = registration.Valis(
        src_dir=slide_src_dir, 
        dst_dir=pair_results_dir, 
        img_list=img_list,           # <-- CRUCIAL: On force VALIS à n'utiliser que ces 2 lames
        reference_img_f=hes_path     # <-- CRUCIAL: On définit le HES comme référence
    )
    
    # Lancer l'enregistrement
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    
    # registered_slide_dst_dir = os.path.join(pair_results_dir, "registered_slides")
    # registrar.warp_and_save_slides(registered_slide_dst_dir)

print("\n✅ Traitement terminé !")