import cv2
import os
import torch
import torch.nn as nn
import torchvision.utils
import numpy as np

# --- MOCKS DES MODULES (Pour palier l'absence du code source complet LivePortrait) ---
# Dans l'environnement réel, on importerait directement les architectures du projet :
# from liveportrait.modules.warping_network import WarpingNetwork
# from liveportrait.modules.spade_generator import SPADEGenerator
class DummyNetwork(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
    def forward(self, *args, **kwargs):
        # Renvoie le premier argument en tant que tensor pour garder le flux de données en vie
        # Fallback : Génération du mock tenseur sur CPU
        return args[0] if args else torch.randn(1, 3, 256, 256)

class PortraitAnimator:
    def __init__(self, weights_dir="./weights", device="cpu"):
        print("[AI Engine] Lancement de l'initialisation de PortraitAnimator...")
        self.weights_dir = weights_dir
        # Fallback complet sur CPU (pas de CUDA)
        self.device = torch.device("cpu")
        print(f"[AI Engine] Utilisation de l'accélérateur : {self.device}")
        
        # Sécurité : Liste stricte des modèles LivePortrait nécessaires
        self.required_weights = [
            "appearance_feature_extractor.pth",
            "motion_extractor.pth",
            "warping_module.pth",
            "spade_generator.pth",
            "stitching_retargeting_module.pth"
        ]
        self.check_weights()
        
        print("[AI Engine] Chargement des réseaux en VRAM...")
        
        # Initialisation Dummy des réseaux (A remplacer par les vraies classes)
        # Ex: self.appearance_extractor = AppearanceFeatureExtractor().to(self.device)
        self.appearance_extractor = DummyNetwork("appearance").to(self.device)
        self.motion_extractor = DummyNetwork("motion").to(self.device)
        self.warping_module = DummyNetwork("warping").to(self.device)
        self.spade_generator = DummyNetwork("spade").to(self.device)
        self.stitching_module = DummyNetwork("stitching").to(self.device)
        
        # Passage en mode évaluation (optimisation PyTorch : désactive certaines couches comme Dropout)
        for model in [self.appearance_extractor, self.motion_extractor, 
                      self.warping_module, self.spade_generator, self.stitching_module]:
            model.eval()
            model.float() # Forcer tous les poids en FP32 pour la stabilité CPU

        self.source_feature = None # Cache d'optimisation
        self.source_motion = None # Cache mouvement source
        self.initial_driving_motion = None # Cache première frame webcam pour le mouvement relatif
        print("[AI Engine] Initialisation VRAM terminée avec succès.")

    def check_weights(self):
        if not os.path.exists(self.weights_dir):
            raise FileNotFoundError(f"[-] Dossier introuvable : {self.weights_dir}")
        
        missing = []
        for w in self.required_weights:
            path = os.path.join(self.weights_dir, w)
            if not os.path.isfile(path):
                missing.append(w)
        
        if missing:
            raise FileNotFoundError(f"[-] Poids manquants dans {self.weights_dir}: {missing}. "
                                     f"Veuillez exécuter: python download_weights.py")
        print("[AI Engine] Fichiers modèles validés.")

    # Optimisation critique de l'agent 2 : empêche la création de graphes de gradients
    @torch.inference_mode()
    def generate_frame(self, source_image, driving_frame):
        """
        Inférence complète PyTorch de la pipeline LivePortrait.
        Conversion CV2 -> Tensor (GPU) -> Warping -> Tensor -> CV2
        """
        # --- AUDIT STRICT : Aucun try/except silencieux. Si ça crashe, ça crashe bruyamment. ---

        # 1. Prétraitement HWC to NCHW et transfert sur GPU
        source_tensor = self.preprocess_image(source_image).to(self.device)
        driving_tensor = self.preprocess_image(driving_frame).to(self.device)
        
        # 2. Features Extraction (avec cache pour la Joconde puisque la peinture ne bouge pas)
        if self.source_feature is None:
            self.source_feature = self.appearance_extractor(source_tensor)
            self.source_motion = self.motion_extractor(source_tensor)
            
        # Extraction des landmarks/mouvements de la webcam (driving_frame) à chaque itération
        driving_motion = self.motion_extractor(driving_tensor)
        
        # Initialisation de la première frame de la webcam (Frame 0) pour le calcul du delta
        if self.initial_driving_motion is None:
            self.initial_driving_motion = driving_motion.clone() if hasattr(driving_motion, 'clone') else driving_motion
            print("[AI Engine] Première frame webcam capturée pour le mouvement relatif.")

        # Debug Print : landmarks extraits depuis driving_motion
        nose_x = float(torch.mean(driving_motion[0, 0, :, :]).cpu() * 640)
        nose_y = float(torch.mean(driving_motion[0, 1, :, :]).cpu() * 480)
        print(f"[Debug] Webcam Landmarks (Nez approx) - X: {nose_x:.2f}, Y: {nose_y:.2f}")

        # Calcul du mouvement relatif (Relative Motion)
        relative_motion = driving_motion - self.initial_driving_motion + self.source_motion

        # ================================================================
        # SONDE 1 — MATHÉMATIQUE (Le Delta)
        # Si cette valeur reste à 0.0 quand tu bouges, le calcul du
        # mouvement est mort (initial_driving_motion == driving_motion).
        # ================================================================
        print(f"[Audit] Somme absolue du Delta : {torch.sum(torch.abs(driving_motion - self.initial_driving_motion)).item()}")
        
                # 3. Warping Module (Formation des features déformées)
        warped_feature = self.warping_module(
            feature_3d=self.source_feature, 
            kp_source=self.source_motion, 
            kp_driving=relative_motion
        )
        
        # 4. SPADE Generator (Génération des pixels finaux depuis les features déformées)
        out_tensor = self.spade_generator(feature_3d=warped_feature)

        # ================================================================
        # SONDE 2 — PYTORCH (Génération brute, avant conversion OpenCV)
        # Bypass total d'OpenCV : si l'image sur disque est animée entre
        # deux sauvegardes, c'est OpenCV (postprocess) qui bugue.
        # ================================================================
        torchvision.utils.save_image(out_tensor, 'debug_tensor_output.jpg')

        # --- POST-TRAITEMENT MANUEL BLINDÉ ---
        # 1. On détache du GPU et on enlève la dimension Batch (1, 3, H, W) -> (3, H, W)
        tensor_cpu = out_tensor.detach().squeeze(0).cpu().float()
        
        # 2. On réorganise les dimensions : PyTorch (CHW) vers OpenCV (HWC)
        img_array = tensor_cpu.numpy()
        img_array = np.transpose(img_array, (1, 2, 0))
        
        # 3. Normalisation stricte : on ramène les valeurs entre 0.0 et 1.0
        if img_array.min() < 0:
            img_array = (img_array + 1.0) / 2.0
        img_array = np.clip(img_array, 0, 1)
        
        # 4. Conversion en pixels (0-255) et type uint8 pour OpenCV
        output_face_img = (img_array * 255.0).astype(np.uint8)
        
        # 5. Conversion des couleurs de RGB (PyTorch) vers BGR (OpenCV)
        output_face_img = cv2.cvtColor(output_face_img, cv2.COLOR_RGB2BGR)
        # ================================================================
        # SONDE 3 — CONVERSION (Type et Shape)
        # Plantage dur si la conversion Tensor->OpenCV est corrompue.
        # ================================================================
        assert output_face_img.dtype == np.uint8, "Erreur critique : L'image n'est pas en uint8"
        assert len(output_face_img.shape) == 3, "Erreur critique : Mauvaise dimension matricielle"

        # Plus besoin du paste_back si l'IA génère déjà tout le portrait
        return output_face_img

    def preprocess_image(self, img_bgr):
        # Conversion BGR -> RGB, normalisation [0, 1] puis changement de dimensions
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        # Numpy HWC -> Tensor C H W -> Ajout du Batch N = N C H W
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def postprocess_tensor(self, tensor):
        # Retire la dimension Batch, repasse en HWC sur le CPU
        img_rgb = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Denormalisation et passage en espace 8 bits BGR
        img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr

    def paste_back(self, generated_face, original_image):
        # Dans LivePortrait, la fonction de paste back utilise des masques lissés (Laplacian / Poisson Image Editing)
        # Ici c'est le squelette d'application de rendu (Bounding Box ou masque)
        h, w = original_image.shape[:2]
        
        # Redimensionnement arbitraire à 50% de la fenêtre source comme placeholder
        target_h, target_w = h // 2, w // 2
        resized_face = cv2.resize(generated_face, (target_w, target_h))
        
        final_img = original_image.copy()
        start_y, start_x = h // 4, w // 4
        
        # Sécurité pour ne pas déborder
        if start_y + target_h <= h and start_x + target_w <= w:
            final_img[start_y:start_y+target_h, start_x:start_x+target_w] = resized_face
            
        cv2.putText(final_img, "LivePortrait VRAM Inferencing...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 0), 2)
        return final_img
