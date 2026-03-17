import cv2
import os
import sys
import torch
import torch.nn as nn
import torchvision.utils
import numpy as np
import yaml
from collections import OrderedDict

# --- IMPORTS DES VRAIES ARCHITECTURES LIVEPORTRAIT ---
# On ajoute le chemin source LivePortrait au sys.path pour résoudre les imports des modules
_lp_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveportrait_src", "src")
sys.path.insert(0, _lp_src_dir)

from modules.appearance_feature_extractor import AppearanceFeatureExtractor
from modules.motion_extractor import MotionExtractor
from modules.warping_network import WarpingNetwork
from modules.spade_generator import SPADEDecoder
from modules.stitching_retargeting_network import StitchingRetargetingNetwork
from utils.camera import headpose_pred_to_degree, get_rotation_matrix


def _remove_ddp_key(state_dict):
    """Nettoie les clés 'module.' des checkpoints entraînés en DataParallel."""
    new_sd = OrderedDict()
    for key in state_dict.keys():
        new_sd[key.replace('module.', '')] = state_dict[key]
    return new_sd


def load_model(ckpt_path, model_config, device, model_type):
    """
    Instancie et charge un module LivePortrait depuis son checkpoint.
    Copie locale de la logique de liveportrait_src/src/utils/helper.py::load_model
    pour éviter les problèmes d'imports relatifs.
    """
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(_remove_ddp_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(_remove_ddp_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(_remove_ddp_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {'stitching': stitcher, 'lip': retargetor_lip, 'eye': retargetor_eye}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.eval()
    return model


class PortraitAnimator:
    def __init__(self, weights_dir="./weights", device="cuda"):
        print("[AI Engine] Lancement de l'initialisation de PortraitAnimator...")
        self.weights_dir = weights_dir

        # GPU déverrouillé : utilise CUDA si disponible, sinon CPU
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
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
        
        # --- CHARGEMENT DES VRAIS RÉSEAUX LIVEPORTRAIT ---
        print("[AI Engine] Chargement des réseaux en VRAM...")

        # Chargement du fichier de configuration des architectures (hyperparamètres)
        models_config_path = os.path.join(_lp_src_dir, "config", "models.yaml")
        model_config = yaml.load(open(models_config_path, 'r'), Loader=yaml.SafeLoader)

        # F — Appearance Feature Extractor (extrait le volume 3D de features de la Joconde)
        self.appearance_extractor = load_model(
            os.path.join(self.weights_dir, "appearance_feature_extractor.pth"),
            model_config, self.device, 'appearance_feature_extractor'
        )
        print("[AI Engine] ✓ Appearance Feature Extractor chargé.")

        # M — Motion Extractor (extrait pitch/yaw/roll/kp/exp/scale/t)
        self.motion_extractor = load_model(
            os.path.join(self.weights_dir, "motion_extractor.pth"),
            model_config, self.device, 'motion_extractor'
        )
        print("[AI Engine] ✓ Motion Extractor chargé.")

        # W — Warping Network (déforme le volume 3D selon les keypoints)
        self.warping_module = load_model(
            os.path.join(self.weights_dir, "warping_module.pth"),
            model_config, self.device, 'warping_module'
        )
        print("[AI Engine] ✓ Warping Module chargé.")

        # G — SPADE Generator (génère l'image finale à partir des features déformées)
        self.spade_generator = load_model(
            os.path.join(self.weights_dir, "spade_generator.pth"),
            model_config, self.device, 'spade_generator'
        )
        print("[AI Engine] ✓ SPADE Generator chargé.")

        # S — Stitching & Retargeting Module (dict de 3 sous-réseaux : stitching, lip, eye)
        self.stitching_module = load_model(
            os.path.join(self.weights_dir, "stitching_retargeting_module.pth"),
            model_config, self.device, 'stitching_retargeting_module'
        )
        print("[AI Engine] ✓ Stitching/Retargeting Module chargé.")

        # Note : load_model() appelle déjà .eval() sur chaque réseau.
        # Le stitching_module est un dict {'stitching', 'lip', 'eye'} — déjà en eval.

        self.source_feature = None       # Cache d'optimisation pour le volume 3D de la Joconde
        self.source_kp_info = None       # Cache des kp_info de la source
        self.source_kp = None            # Cache des keypoints transformés de la source (BxNx3)
        self.initial_driving_kp_info = None  # Cache première frame webcam pour le mouvement relatif
        self.initial_driving_kp = None   # Keypoints transformés de la frame 0 de la webcam
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

    def get_kp_info(self, x):
        """
        Extraction des keypoints via le Motion Extractor.
        x: Bx3xHxW tensor, normalisé 0~1
        Retourne un dict raffiné : {kp (BxNx3), pitch, yaw, roll, t, exp, scale}
        """
        with torch.no_grad():
            kp_info = self.motion_extractor(x)
            # Float tous les tenseurs (sécurité half-precision)
            for k, v in kp_info.items():
                if isinstance(v, torch.Tensor):
                    kp_info[k] = v.float()

        # Raffinement : conversion des angles en degrés et reshape des keypoints
        bs = kp_info['kp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
        kp_info['yaw']   = headpose_pred_to_degree(kp_info['yaw'])[:, None]    # Bx1
        kp_info['roll']  = headpose_pred_to_degree(kp_info['roll'])[:, None]   # Bx1
        kp_info['kp']    = kp_info['kp'].reshape(bs, -1, 3)                   # BxNx3
        kp_info['exp']   = kp_info['exp'].reshape(bs, -1, 3)                  # BxNx3

        return kp_info

    def transform_keypoint(self, kp_info):
        """
        Transforme les keypoints canoniques selon la pose, l'échelle et l'expression.
        Eqn.2 du paper : s * (R * x_c + exp) + t
        Retourne: BxNx3
        """
        kp = kp_info['kp']       # (bs, N, 3)
        pitch = kp_info['pitch'] # (bs, 1)
        yaw   = kp_info['yaw']
        roll  = kp_info['roll']
        t     = kp_info['t']
        exp   = kp_info['exp']   # (bs, N, 3)
        scale = kp_info['scale']

        bs = kp.shape[0]
        num_kp = kp.shape[1]

        rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (bs, 3, 3)

        # Eqn.2: s * (R * x_c + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)
        kp_transformed *= scale[..., None]                     # (bs, N, 3) * (bs, 1, 1)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]          # Appliquer tx, ty (pas tz)

        return kp_transformed

    # Optimisation critique : empêche la création de graphes de gradients
    @torch.inference_mode()
    def generate_frame(self, source_image, driving_frame):
        """
        Inférence complète PyTorch de la pipeline LivePortrait.
        Conversion CV2 -> Tensor (GPU) -> Motion -> Warping -> SPADE -> Tensor -> CV2
        """
        # --- AUDIT STRICT : Aucun try/except silencieux. Si ça crashe, ça crashe bruyamment. ---

        # 1. Prétraitement HWC to NCHW et transfert sur GPU
        source_tensor = self.preprocess_image(source_image).to(self.device)
        driving_tensor = self.preprocess_image(driving_frame).to(self.device)
        
        # 2. Features Extraction (avec cache pour la Joconde puisque la peinture ne bouge pas)
        if self.source_feature is None:
            self.source_feature = self.appearance_extractor(source_tensor)  # Bx32x16x64x64
            self.source_kp_info = self.get_kp_info(source_tensor)
            self.source_kp = self.transform_keypoint(self.source_kp_info)  # BxNx3
            print(f"[AI Engine] Source feature shape: {self.source_feature.shape}")
            print(f"[AI Engine] Source kp shape: {self.source_kp.shape}")

        # Extraction des keypoints de la webcam (driving_frame) à chaque itération
        driving_kp_info = self.get_kp_info(driving_tensor)
        driving_kp = self.transform_keypoint(driving_kp_info)  # BxNx3

        # Initialisation de la première frame de la webcam (Frame 0) pour le calcul du delta
        if self.initial_driving_kp_info is None:
            # Deep copy du dict de kp_info pour figer la frame 0
            self.initial_driving_kp_info = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                             for k, v in driving_kp_info.items()}
            self.initial_driving_kp = driving_kp.clone()
            print("[AI Engine] Première frame webcam capturée pour le mouvement relatif.")

        # Debug Print : pose extraite (pitch/yaw/roll en degrés)
        pitch_val = float(driving_kp_info['pitch'].cpu())
        yaw_val   = float(driving_kp_info['yaw'].cpu())
        print(f"[Debug] Webcam Pose - Pitch: {pitch_val:.2f}°, Yaw: {yaw_val:.2f}°")

        # Calcul du mouvement relatif (Relative Motion)
        # kp_relative = kp_source + (kp_driving_current - kp_driving_initial)
        kp_relative = self.source_kp + (driving_kp - self.initial_driving_kp)

        # ================================================================
        # SONDE 1 — MATHÉMATIQUE (Le Delta)
        # Si cette valeur reste à 0.0 quand tu bouges, le calcul du
        # mouvement est mort (initial_driving_kp == driving_kp).
        # ================================================================
        delta = driving_kp - self.initial_driving_kp
        print(f"[Audit] Somme absolue du Delta : {torch.sum(torch.abs(delta)).item()}")
        
        # 3. Warping Module (Formation des features déformées)
        # WarpingNetwork.forward(feature_3d, kp_driving, kp_source) -> dict avec clé 'out'
        ret_dct = self.warping_module(
            self.source_feature, 
            kp_driving=kp_relative, 
            kp_source=self.source_kp
        )
        
        # 4. SPADE Generator (Génération des pixels finaux)
        # SPADEDecoder.forward(feature) -> tensor Bx3xHxW (sigmoïd, donc déjà entre 0 et 1)
        out_tensor = self.spade_generator(feature=ret_dct['out'])

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
        # Resize à 256x256 (taille d'entrée standard LivePortrait)
        img_rgb = cv2.resize(img_rgb, (256, 256))
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
