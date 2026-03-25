import os, random, numpy as np, cv2, scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def crop(allfocus, fs, gt, contour, depth):
    image_size = allfocus.shape[0]
    crop_size = image_size - 20
    ix = 2 * random.randint(0, 9)
    iy = 2 * random.randint(0, 9)

    na = allfocus[ix:ix+crop_size, iy:iy+crop_size]
    nf = fs[ix:ix+crop_size, iy:iy+crop_size]
    ng = gt[ix:ix+crop_size, iy:iy+crop_size]
    nc = contour[ix:ix+crop_size, iy:iy+crop_size]
    nd = depth[ix:ix+crop_size, iy:iy+crop_size]

    na = cv2.resize(na, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    nf = cv2.resize(nf, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    ng = cv2.resize(ng, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    nc = cv2.resize(nc, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    nd = cv2.resize(nd, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return na, nf, ng, nc, nd

class LFDataset(Dataset):
    def __init__(self, location=None, train=True, crop=True, image_size=224):
        self.location = location
        self.train = train
        self.crop_flag = crop
        self.image_size = image_size

        self.img_list = os.listdir(os.path.join(self.location, 'allfocus'))
        self.img_list.sort()
        self.num = len(self.img_list)

    def __len__(self): return self.num

    def _load_allfocus(self, name):
        img = Image.open(os.path.join(self.location, 'allfocus', name)).convert('RGB')
        img = img.resize((self.image_size, self.image_size))
        return np.asarray(img)

    def _load_focalstack(self, stem):
        mat_path = os.path.join(self.location, 'mat', stem + '.mat')
        fs = sio.loadmat(mat_path)['img'].astype(np.float32)
        fs = cv2.resize(fs, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return fs

    def _load_mask(self, folder, stem):
        p = os.path.join(self.location, folder, stem + '.png')
        m = Image.open(p).convert('L').resize((self.image_size, self.image_size))
        return np.asarray(m)

    def _load_depth(self, stem):
        cand = [
            os.path.join(self.location, 'depth', stem + '.png'),
            os.path.join(self.location, 'depth', stem + '.tiff'),
            os.path.join(self.location, 'depth', stem + '.exr'),
            os.path.join(self.location, 'depth', stem + '.jpg'),
            os.path.join(self.location, 'depth', stem + '.npy'),
            os.path.join(self.location, 'depth', stem + '.mat'),
        ]
        depth = None
        for p in cand:
            if not os.path.exists(p): continue
            if p.endswith('.npy'):
                depth = np.load(p).astype(np.float32)
            elif p.endswith('.mat'):
                md = sio.loadmat(p)
                for k in ['depth', 'Depth', 'D', 'depth_map', 'd']:
                    if k in md:
                        depth = md[k].astype(np.float32); break
                if depth is None:
                    raise KeyError(f"No depth key found in {p}")
            else:
                im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if im is None: continue
                if im.ndim == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                depth = im.astype(np.float32)
            break
        if depth is None:
            raise FileNotFoundError(f"Depth not found for {stem}")

        # 归一化到 [0,1]
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin + 1e-8)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)

        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return depth  # (H,W) float32 in [0,1]

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        stem = img_name.split('.')[0]

        allfocus = self._load_allfocus(img_name)          # (H,W,3) uint8
        focal    = self._load_focalstack(stem)            # (H,W,3*S) float32
        depth    = self._load_depth(stem)                 # (H,W)     float32 [0,1]

        if self.train:
            GT      = self._load_mask('GT', stem)         # (H,W)
            contour = self._load_mask('contour', stem)    # (H,W)

            if self.crop_flag:
                allfocus, focal, GT, contour, depth = crop(allfocus, focal, GT, contour, depth)

            allfocus = transforms.ToTensor()(allfocus)     # (3,H,W)
            focal    = transforms.ToTensor()(focal)        # (3*S,H,W)
            GT       = transforms.ToTensor()(GT[..., None])        # (1,H,W)
            contour  = transforms.ToTensor()(contour[..., None])   # (1,H,W)
            depth    = transforms.ToTensor()(depth[..., None])     # (1,H,W)

            return allfocus, focal, depth, GT, contour, img_name
        else:
            GT = self._load_mask('GT', stem)
            allfocus = transforms.ToTensor()(allfocus)
            focal    = transforms.ToTensor()(focal)
            depth    = transforms.ToTensor()(depth[..., None])
            GT       = transforms.ToTensor()(GT)
            return allfocus, focal, depth, GT, img_name
