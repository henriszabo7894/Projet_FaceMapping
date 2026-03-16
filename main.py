"""
Agent 3: I/O & Real-Time Pipeline (Le Chef d'Orchestre)
Gère le flux de données Temps Réel de la webcam vers le moteur IA 
et calcule les Performances (FPS).
"""

import cv2
import time
import os
from ai_engine import PortraitAnimator

def main():
    print("=== Artimir 2.0 - Lancement du Pipeline Temps Réel (Agent 3) ===")
    
    # 1. Instanciation du Cerveau (Agent 2)
    # Chargement en VRAM du modèle unique
    try:
        animator = PortraitAnimator()
    except Exception as e:
        print(f"[-] Erreur critique lors du chargement de l'Agent 2 : {e}")
        return

    # 2. Chargement de l'Image Source (La Joconde)
    source_img_path = "./joconde.jpg"
    
    # Création d'une image vierge si Joconde non trouvée, pour éviter que le test plante.
    if not os.path.exists(source_img_path):
        import numpy as np
        print(f"[!] Attention : '{source_img_path}' introuvable. Création d'un mock.")
        source_image = np.zeros((512, 512, 3), dtype=np.uint8)
        # Remplir en marron/gris pour simuler le tableau
        source_image[:] = (70, 100, 150) # BGR
        cv2.putText(source_image, "Joconde.jpg_Manquante", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        source_image = cv2.imread(source_img_path)
        print(f"[+] Source chargée : {source_img_path} (Shape: {source_image.shape})")

    # 3. Initialisation du Flux Input Webcam (Idéalement /dev/video0)
    print("\n[*] Initialisation de la capture Webcam...")
    cap = cv2.VideoCapture(0)
    
    # Règle d'or: Configurer la cam pour limiter l'I/O bloquant
    # Exemple 640x480 @ 30 FPS pour un bon compromis qualité/RT
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[-] Erreur : Impossible d'ouvrir la webcam.")
        return

    # 4. Boucle Principale
    print("[+] Début du Streaming (Appuyez sur 'q' ou 'ESC' pour quitter).")
    prev_time = time.time()
    
    try:
        while True:
            # --- PHASE I/O : Acquisition ---
            ret, driving_frame = cap.read()
            if not ret:
                print("[-] Fin du flux webcam.")
                break
                
            # Miroir l'image de la webcam pour un comportement naturel
            driving_frame = cv2.flip(driving_frame, 1)

            # --- PHASE IA : Génération (Le Cerveau) ---
            # Bloquant par nature, d'où l'importance de CUDA
            t_inference_start = time.time()
            generated_img = animator.generate_frame(source_image, driving_frame)
            t_inference_end = time.time()

            # --- PHASE PERFORMANCES : Monitorer la RTX 5080 ---
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Temps brut d'inférence GPU en ms (Network Forward Pass + Tensors Cpy)
            latency_ms = (t_inference_end - t_inference_start) * 1000

            # --- PHASE I/O : Rendu & GUI ---
            # Affichage de l'UI (Incrustations textuelles performantes)
            text_color = (0, 255, 0) if fps > 20 else (0, 0, 255) # Vert si >20fps, sinon rouge
            cv2.putText(generated_img, f"Artimir 2.0 (RTX 5080)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(generated_img, f"FPS: {int(fps)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            cv2.putText(generated_img, f"Inference Latency: {int(latency_ms)} ms", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Assemblage côte à côte optionnel pour visualisation complète (Webcam + Resultat)
            # Resize Webcam to match Source for stacking
            h_g, w_g = generated_img.shape[:2]
            driving_resized = cv2.resize(driving_frame, (w_g, h_g))
            combined_view = cv2.hconcat([driving_resized, generated_img])

            cv2.imshow("Artimir 2.0 - Live Portrait Animation", combined_view)

            # --- Sortie ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 27 = Echap
                break
                
    except KeyboardInterrupt:
        print("\n[!] Interruption manuelle.")
    finally:
        # 5. Nettoyage mémoire proper (Clean Architecture)
        print("[*] Fermeture I/O et VRAM...")
        cap.release()
        cv2.destroyAllWindows()
        # Vider le cache CUDA peut être utile pour de grosses applis
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[+] Terminé.")

if __name__ == "__main__":
    main()
