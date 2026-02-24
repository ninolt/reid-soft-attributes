import os
import zipfile

import gdown
import numpy as np
import requests
import scipy.io
import torch
from torch.utils.data import DataLoader
from torchreid.reid.data import ImageDataManager, register_image_dataset
from torchreid.reid.data.datasets.image import Market1501
from torchreid.reid.data.transforms import build_transforms

from src.config import CFG
from src.utils import get_logger

logger = get_logger("reid.dataset")

_MAT_COLS = [
    "image_index", "gender", "hair", "up", "down", "clothes",
    "hat", "backpack", "bag", "handbag", "age",
    "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen",
    "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue", "downgreen", "downbrown",
]
_UP_COLOR = _MAT_COLS[11:19]
_DOWN_COLOR = _MAT_COLS[19:28]


def load_market1501_attributes(attr_mat_path, split="train"):
    """Load Market-1501 attributes from the official .mat file. Returns {pid: {attributes}}."""
    if not os.path.exists(attr_mat_path):
        return {}

    def _s1d(a):
        """Convert MATLAB array to 1D numpy int64 array."""
        a = np.array(a).squeeze()
        if a.dtype == object:
            return np.array(
                [int(np.squeeze(np.array(v))) for v in a.reshape(-1)],
                dtype=np.int64,
            )
        if a.ndim == 0:
            return np.array([int(a)], dtype=np.int64)
        return a.astype(np.int64).reshape(-1)

    def _bin(x):
        return 0 if x == 1 else 1 if x == 2 else 0

    def _cls(x, k):
        return 0 if x < 1 else k - 1 if x > k else x - 1

    def _onehot(flags):
        return int(np.argmax([_bin(int(v)) for v in flags]))

    try:
        mat_data = scipy.io.loadmat(attr_mat_path)
        if "market_attribute" in mat_data:
            d = mat_data["market_attribute"].reshape(-1)[0][split].reshape(-1)[0]
        else:
            # For gallery_market.mat or other formats
            key = [k for k in mat_data.keys() if not k.startswith("__")][0]
            d = mat_data[key].reshape(-1)[0]

        mat = np.stack([_s1d(d[c]) for c in _MAT_COLS], axis=1)
        if mat.shape[1] != len(_MAT_COLS):
            logger.warning("Expected %d columns, got %d", len(_MAT_COLS), mat.shape[1])
            return {}

        id2t = {}
        for row in mat:
            rec = dict(zip(_MAT_COLS, [int(x) for x in row]))
            pid = int(rec["image_index"])
            id2t[pid] = {
                "gender": _bin(rec["gender"]),
                "hair": _bin(rec["hair"]),
                "sleeve": _bin(rec["up"]),
                "down_length": _bin(rec["down"]),
                "lower_type": _bin(rec["clothes"]),
                "hat": _bin(rec["hat"]),
                "backpack": _bin(rec["backpack"]),
                "bag": _bin(rec["bag"]),
                "handbag": _bin(rec["handbag"]),
                "age": _cls(rec["age"], CFG["dataset"]["age_num_classes"]),
                "upper_color": _onehot([rec[c] for c in _UP_COLOR]),
                "lower_color": _onehot([rec[c] for c in _DOWN_COLOR]),
            }
        return id2t
    except Exception as e:
        logger.warning("Could not load attributes from %s: %s", attr_mat_path, e)
        return {}


def download_file(url, filename):
    if os.path.exists(filename):
        return filename

    logger.info("Downloading %s...", url)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=CFG["dataset"]["download_chunk_size"]):
            f.write(chunk)
    return filename


class Market1501WithAttributes(Market1501):
    MARKET_ATTR_URL: str = CFG["dataset"]["market_attr_url"]
    GALLERY_ATTR_URL: str = CFG["dataset"]["gallery_attr_url"]
    
    def __init__(self, root="datasets", transform=None, **kwargs):
        self.root = os.path.abspath(os.path.expanduser(root))
        dataset_dir = os.path.join(self.root, "market1501")
        
        if not os.path.exists(os.path.join(dataset_dir, "Market-1501-v15.09.15")):
            logger.info("Downloading Market-1501 via GDrive...")
            os.makedirs(self.root, exist_ok=True)
            zip_path = os.path.join(self.root, "market1501.zip")
            gdown.download(id="0B8-rUzbwVRk0c054eEozWG9COHM", output=zip_path)
            
            with zipfile.ZipFile(zip_path, "r") as z:
                os.makedirs(dataset_dir, exist_ok=True)
                z.extractall(dataset_dir)
            os.remove(zip_path)

        super().__init__(root=root, transform=transform, **kwargs)

        market_attr_path = os.path.join(self.root, "market_attribute.mat")
        gallery_attr_path = os.path.join(self.root, "gallery_market.mat")
        
        download_file(self.MARKET_ATTR_URL, market_attr_path)
        download_file(self.GALLERY_ATTR_URL, gallery_attr_path)

        self.train_attrs = load_market1501_attributes(market_attr_path, "train")
        self.test_attrs = load_market1501_attributes(gallery_attr_path, "test")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        
        img = item["img"]
        pid = item["pid"]
        camid = item["camid"]
        impath = item["impath"]
        attrs = self.get_attributes(pid)
        
        return img, pid, camid, impath, attrs

    def get_attributes(self, pid: int) -> dict:
        """Get attributes for a PID from appropriate split.
        
        Args:
            pid: Person ID
        
        Returns:
            Dictionary with "gender", "hair", "age" or empty dict if not found.
        """
        if pid in self.train_attrs:
            return self.train_attrs[pid]
        if pid in self.test_attrs:
            return self.test_attrs[pid]
        return {}


register_image_dataset("market1501_attr", Market1501WithAttributes)


def get_train_transform():
    """Get training transform pipeline using torchreid's build_transforms.
    
    Applied to individual PIL images (e.g., for IQA).
    Matches the transforms used by ImageDataManager for training.
    """
    _ds = CFG["dataset"]
    return build_transforms(
        is_train=True,
        height=_ds["height"],
        width=_ds["width"],
        transforms=_ds["transforms"],
    )


def _collate_with_attributes(batch):
    """Collate function that stacks attribute dictionaries from batch."""
    images, labels, camids, img_paths, attrs_list = zip(*batch)
    
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    camids_tensor = torch.tensor(camids, dtype=torch.long)
    
    # Stack attributes: collect all keys and create batch tensors
    attrs_batch = {}
    if any(attrs_list):
        for key in set(k for attrs in attrs_list for k in attrs):
            attrs_batch[key] = torch.tensor(
                [int(attrs.get(key, 0)) for attrs in attrs_list],
                dtype=torch.long,
            )
    
    return images_tensor, labels_tensor, camids_tensor, list(img_paths), attrs_batch


def build_dataloaders(data_dir: str):
    """Build dataloaders using ImageDataManager with RandomIdentitySampler."""
    _ds = CFG["dataset"]
    manager = ImageDataManager(
        root=data_dir,
        sources=_ds["name"],
        targets=_ds["name"],
        height=_ds["height"],
        width=_ds["width"],
        batch_size_train=_ds["batch_size_train"],
        batch_size_test=_ds["batch_size_test"],
        train_sampler=_ds["train_sampler"],
        transforms=_ds["transforms"],
    )
    
    train_loader = DataLoader(
        dataset=manager.train_loader.dataset,
        batch_size=manager.train_loader.batch_size,
        sampler=manager.train_loader.sampler,
        num_workers=manager.train_loader.num_workers,
        collate_fn=_collate_with_attributes,
        pin_memory=CFG["dataset"]["pin_memory"],
    )
    
    target_name: str = CFG["dataset"]["name"]
    return (
        train_loader,
        manager.test_loader[target_name]["gallery"],
        manager.test_loader[target_name]["query"],
        manager.num_train_pids,
    )

