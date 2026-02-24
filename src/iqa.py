import os

import numpy as np
import torch
import torchreid
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial.distance import pdist

from src.config import CFG
from src.dataset import get_train_transform
from src.utils import get_logger

logger = get_logger("reid.iqa")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logger.info("Device used: %s", device)

model = torchreid.models.build_model(
    name=CFG["iqa"]["backbone"],
    num_classes=CFG["iqa"]["num_classes"],
    pretrained=True
)

model.eval().to(device)

transform = get_train_transform()[0]

def calculate_q_score(features):
    # Mean pairwise distance between embeddings: higher distance -> lower quality.
    return np.mean(pdist(features, metric="euclidean"))


def get_iqa(img_path: str) -> float:
    original_image = Image.open(img_path)

    perturbed_images = []

    perturbed_images += lightning_perturbations(original_image)
    perturbed_images += blur_perturbations(original_image)
    perturbed_images += occlusion_perturbations(original_image)
    perturbed_images += background_perturbations(original_image)
    perturbed_images += noise_perturbations(original_image)
    tensors_list = [transform(original_image)]
    tensors_list += [transform(perturbed_image) for perturbed_image in perturbed_images]
    batch_tensor = torch.stack(tensors_list).to(device)
    with torch.no_grad():
        embeddings = model(batch_tensor)
        flatten_embeddings = torch.flatten(embeddings, start_dim=1)

    embeddings_np = flatten_embeddings.cpu().numpy()

    iqa_score = calculate_q_score(embeddings_np)

    return float(iqa_score)


def lightning_perturbations(img: Image.Image) -> list[Image.Image]:
    # V-channel shifts in HSV to simulate lighting changes.
    h, s, v= img.convert("HSV").split()
    v_array = np.array(v, dtype=np.int16)
   
    shifts: list[int] = CFG["iqa"]["perturbations"]["lightning_shifts"]
    
    variants = []
    
    for shift in shifts:
        new_v = v_array + shift
        
        new_v = np.clip(new_v, 0, 255).astype(np.uint8)
        
        new_v_img = Image.fromarray(new_v, mode="L")
        new_hsv = Image.merge("HSV", (h, s, new_v_img))
        
        variants.append(new_hsv.convert('RGB'))
        
    return variants


def blur_perturbations(img: Image.Image) -> list[Image.Image]:
    variants = []

    sigmas: list[float] = CFG["iqa"]["perturbations"]["blur_sigmas"]

    for sigma in sigmas:
        variants.append(img.copy().filter(ImageFilter.GaussianBlur(radius=sigma)))

    return variants


def occlusion_perturbations(img: Image.Image) -> list[Image.Image]:
    width, height = img.size
    _pad_factor: float = CFG["iqa"]["perturbations"]["occlusion_pad_factor"]
    pad_w = width * _pad_factor
    pad_h = height * _pad_factor
    
    percentages: list[float] = CFG["iqa"]["perturbations"]["occlusion_percentages"]
    
    perturbed_images = []

    for side in range(4):
        for p in percentages:
            img_copy = img.copy()
            draw = ImageDraw.Draw(img_copy)
            
            # Provide a safe default so coords is always defined
            coords = [pad_w, pad_h, pad_w, pad_h]
            if side == 0:
                coords = [pad_w, pad_h, width - pad_w, pad_h + (height * p)]
            elif side == 1:
                coords = [pad_w, height - pad_h - (height * p), width - pad_w, height - pad_h]
            elif side == 2:
                coords = [pad_w, pad_h, pad_w + (width * p), height - pad_h]
            else:
                coords = [width - pad_w - (width * p), pad_h, width - pad_w, height - pad_h]
            
            draw.rectangle(coords, fill="black")
            
            perturbed_images.append(img_copy)
            
    return perturbed_images


def background_perturbations(img: Image.Image) -> list[Image.Image]:
    img = img.convert("RGB")
    width, height = img.size
    
    colors: list[tuple[int, int, int]] = [
        tuple(c) for c in CFG["iqa"]["perturbations"]["background_colors"]
    ]
    scales: list[float] = CFG["iqa"]["perturbations"]["background_scales"]

    perturbed_images = []

    for scale in scales:
        margin_w = (width * (1 - scale)) / 2
        margin_h = (height * (1 - scale)) / 2
        coords = [margin_w, margin_h, width - margin_w, height - margin_h]
        
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse(coords, fill=255)
        
        for color in colors:
            bg = Image.new("RGB", (width, height), color)
            
            perturbed_images.append(Image.composite(img, bg, mask))
            
    return perturbed_images


def noise_perturbations(img: Image.Image) -> list[Image.Image]:
    original_array = np.array(img).astype(float)
    perturbed_images = []

    # Gaussian noise
    _gaussian_scale: float = CFG["iqa"]["perturbations"]["noise_gaussian_scale"]
    noise = np.random.normal(0, _gaussian_scale, original_array.shape)
    gauss_img = np.clip(original_array + noise, 0, 255).astype(np.uint8)
    perturbed_images.append(Image.fromarray(gauss_img))

    # Salt and pepper noise
    _sp_amount: float = CFG["iqa"]["perturbations"]["noise_salt_pepper_amount"]
    sp_array = np.copy(original_array)
    probs = np.random.rand(*sp_array.shape[:2])
    sp_array[probs < (_sp_amount / 2)] = 255  # Salt
    sp_array[(probs >= (_sp_amount / 2)) & (probs < _sp_amount)] = 0  # Pepper
    perturbed_images.append(Image.fromarray(sp_array.astype(np.uint8)))

    # Speckle noise
    _speckle_scale: float = CFG["iqa"]["perturbations"]["noise_speckle_scale"]
    noise = np.random.randn(*original_array.shape) * _speckle_scale
    speckle_img = np.clip(original_array + original_array * noise, 0, 255).astype(np.uint8)
    perturbed_images.append(Image.fromarray(speckle_img))

    return perturbed_images


def compute_iqa_scores(dataset, cache_file=CFG["iqa"]["cache_path"]):
    img_paths = [item[0] for item in dataset.data]
    
    logger.info("Computing IQA scores for %d images...", len(img_paths))
    if os.path.exists(cache_file):
        logger.info("Loading cache from %s", cache_file)
        cache = np.load(cache_file, allow_pickle=True).item()
    else:
        logger.info("No cache found, will create %s", cache_file)
        cache = {}
    
    iqa_scores = []
    computed = 0
    cached = 0
    
    for i, img_path in enumerate(img_paths):
        if (i + 1) % CFG["iqa"]["progress_log_interval"] == 0 or (i + 1) == len(img_paths):
            logger.info("Progress: %d/%d (cached: %d, computed: %d)", i + 1, len(img_paths), cached, computed)
        cache_key = os.path.basename(img_path)
        if cache_key in cache:
            iqa_score = cache[cache_key]
            cached += 1
        else:
            iqa_score = get_iqa(img_path)
            
            cache[cache_key] = iqa_score
            computed += 1
        
        iqa_scores.append(iqa_score)
    logger.info("Saving cache to %s (total entries: %d)", cache_file, len(cache))
    # Save the dict as an object array so it satisfies ArrayLike typing and can be loaded with allow_pickle=True
    np.save(cache_file, np.array(cache, dtype=object))
    
    return iqa_scores
