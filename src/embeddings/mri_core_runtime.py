import os, sys, math, numpy as np, torch, cv2, traceback
from monai.transforms import Compose, RandAffine, RandGaussianNoise, RandBiasField, RandAdjustContrast
from monai.utils import set_determinism

def load_dicom_or_image(path: str) -> np.ndarray:
    try:
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, 'RescaleSlope', 1.0)); inter = float(getattr(ds, 'RescaleIntercept', 0.0))
        arr = arr * slope + inter
        if getattr(ds, 'PhotometricInterpretation', '').upper() == 'MONOCHROME1':
            mn, mx = arr.min(), arr.max(); arr = mx - (arr - mn)
        return arr
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(f'Cannot read image: {path}')
        return img.astype(np.float32)

def to_float01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn: return np.zeros_like(x, np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    if h == w: return img
    s = max(h, w); out = np.zeros((s, s), img.dtype); t = (s - h)//2; l = (s - w)//2
    out[t:t+h, l:l+w] = img; return out

class MRICoreEmbedder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mc = cfg.get('mri_core', {})
        self.image_size = int(mc.get('image_size', 1024))
        self.device = torch.device('cuda' if (mc.get('device','cuda').startswith('cuda') and torch.cuda.is_available()) else 'cpu')
        self.fp16 = (mc.get('dtype','fp16')=='fp16' and self.device.type=='cuda')
        self.model = self._load_mri_core(mc).eval().to(self.device)
        for p in self.model.parameters(): p.requires_grad = False
        ic = cfg.get('image_aug', {})
        self.n_aug = int(ic.get('n_aug_per_sample', 0))
        self.aug_enabled = bool(ic.get('enabled', False)) and self.n_aug>0
        self.policy = self._build_monai_policy(ic)
        self._cache = {}

    def _load_mri_core(self, mc):
        from src.mri_foundation.models.sam import sam_model_registry
        from src.mri_foundation.cfg import parse_args
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0]]
            args = parse_args()
        finally:
            sys.argv = old_argv
        args.devices=[0]; args.num_cls=1; args.image_size=self.image_size
        model = sam_model_registry['vit_b'](
            args=args,
            checkpoint=mc.get('checkpoint', mc.get('ckpt_path', 'checkpoints/mri_foundation.pth')),
            pretrained_sam=True
        )
        return model

    def _build_monai_policy(self, ic: dict):
        rot_deg   = float(ic.get('rot_deg', 7))
        scale_max = float(ic.get('scale_max', 1.05))
        translate_px = float(ic.get('translate_px', 5))
        noise_std = float(ic.get('noise_std', 0.02))
        bias_coeff = float(ic.get('bias_coeff', 0.3))
        gamma_log = float(ic.get('contrast_gamma_log', 0.10))
        tfrac = translate_px / float(self.image_size)
        return Compose([
            RandAffine(prob=0.7, rotate_range=(0.0, 0.0, math.radians(rot_deg)), translate_range=(tfrac, tfrac),
                       scale_range=(scale_max-1.0, scale_max-1.0), mode='bilinear', padding_mode='border'),
            RandBiasField(prob=0.3, coeff_range=(0.0, bias_coeff)),
            RandAdjustContrast(prob=0.3, gamma=(math.exp(-gamma_log), math.exp(gamma_log))),
            RandGaussianNoise(prob=0.3, mean=0.0, std=noise_std),
        ])

    def _prep_tensor(self, img01: np.ndarray) -> torch.Tensor:
        sq = pad_to_square(img01)
        rz = cv2.resize(sq, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img3 = np.stack([rz, rz, rz], axis=0)
        return torch.from_numpy(img3).unsqueeze(0).float()

    @torch.no_grad()
    def embed_original_cached(self, path: str) -> np.ndarray:
        v = self._cache.get(path)
        if v is not None: return v
        img = to_float01(load_dicom_or_image(path))
        x = self._prep_tensor(img).to(self.device)
        if self.device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', enabled=self.fp16):
                feat = self.model.image_encoder(x)
        else:
            feat = self.model.image_encoder(x)
        vec = feat.mean(dim=(2,3)).float().cpu().numpy()[0]
        self._cache[path] = vec
        return vec

    @torch.no_grad()
    def embed_with_aug(self, path: str, n_aug: int, key: str) -> np.ndarray:
        base = self.embed_original_cached(path)[None, :]
        if not self.aug_enabled or n_aug<=0: return base
        img01 = to_float01(load_dicom_or_image(path))
        embs = [base]
        for k in range(n_aug):
            set_determinism(seed=self._aug_seed(key, k))
            sq = pad_to_square(img01)
            rz = cv2.resize(sq, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(rz).unsqueeze(0).float()
            t_aug = self.policy(t)
            ia = t_aug[0].cpu().numpy().astype(np.float32)
            img3 = np.stack([ia, ia, ia], axis=0)
            x = torch.from_numpy(img3).unsqueeze(0).float().to(self.device)
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', enabled=self.fp16):
                    feat = self.model.image_encoder(x)
            else:
                feat = self.model.image_encoder(x)
            vec = feat.mean(dim=(2,3)).float().cpu().numpy()[0]
            embs.append(vec[None, :])
        return np.vstack(embs)

    def _aug_seed(self, key: str, k: int) -> int:
        import hashlib
        s = f"{key}|{k}|{self.cfg.get('image_aug',{}).get('policy','monai')}"
        return int(hashlib.sha1(s.encode()).hexdigest(), 16) % (2**31 - 1)
