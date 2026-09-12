import functools
import sys
import os
import random
import datetime
from pathlib import Path
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image


def collect_ref_images(source: str) -> list[str]:
    p = Path(source)
    if not p.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    exts = {".jpg", ".jpeg", ".png"}
    files = [str(f) for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    files.sort()
    if not files:
        raise FileNotFoundError(f"no images found in source directory: {source}")
    return files


@functools.lru_cache(maxsize=2)
def _load_pipe(model: str, device: str, dtype, with_ip_adapter: bool):
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, use_safetensors=True)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(prompt: str, steps: int = 1, seed: int | None = None, out_dir: str = "out", model: str | None = None, guidance: float = 0.0, height: int = 512, width: int = 512, *, ref_images: list[str] | None = None, ref_scale: float = 0.8) -> str:
    """Generate an image, return absolute path to the PNG on disk."""
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}", file=sys.stderr)
    if device == "cpu":
        print("WARNING: No GPU available, running on CPU - inference will be slow", file=sys.stderr)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {seed}", file=sys.stderr)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    _gen_device = "cpu" if device == "mps" else device
    generator = torch.Generator(device=_gen_device).manual_seed(seed)
    model_explicit = model is not None
    if model is None:
        model = "stabilityai/sd-turbo"
    if ref_images:
        is_sdxl = "sdxl" in model.lower()
        if not is_sdxl:
            if model_explicit:
                raise ValueError(f"IP-Adapter reference images require an SDXL model, got '{model}'")
            print(f"Switching model to stabilityai/sdxl-turbo for IP-Adapter (reference images require SDXL, sd-turbo incompatible)", file=sys.stderr)
            model = "stabilityai/sdxl-turbo"
            if height == 512 and width == 512:
                height = width = 1024
    # validate ref images before loading weights so bad path never triggers adapter load
    if ref_images:
        for p in ref_images:
            if not os.path.exists(p):
                raise FileNotFoundError(f"reference image not found: {p}")
            try:
                with Image.open(p) as _im:
                    _im.verify()
            except FileNotFoundError:
                raise
            except Exception as e:
                # re-check open/convert path to surface unreadable error
                try:
                    Image.open(p).convert("RGB")
                except Exception as e2:
                    raise ValueError(f"unreadable reference image: {p}: {e2}") from e2
                raise ValueError(f"unreadable reference image: {p}: {e}") from e
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    with_ip_adapter = bool(ref_images)
    pipe = _load_pipe(model, device, dtype, with_ip_adapter)
    # ponytail: sdxl-turbo (guidance 0.0 / 1-2 steps) = weaker identity lock for IP-Adapter; upgrade to full SDXL base with guidance ~5 and 30 steps if likeness insufficient
    extra_kwargs: dict = {}
    if ref_images:
        pil_images: list[Image.Image] = []
        for p in ref_images:
            try:
                pil_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                raise ValueError(f"unreadable reference image: {p}: {e}") from e
        n = len(pil_images)
        # reload adapter every call with correct n so stale n/weights from previous call cannot leak (correct over clever; cache still isolates adapter vs text-only pipes)
        pipe.load_ip_adapter(["h94/IP-Adapter"] * n, subfolder=["sdxl_models"] * n, weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"] * n, image_encoder_folder="models/image_encoder")
        pipe.set_ip_adapter_scale([ref_scale] * n)
        extra_kwargs["ip_adapter_image"] = pil_images
    with torch.inference_mode():
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator, height=height, width=width, **extra_kwargs).images[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{seed}.png"
    path = os.path.join(out_dir, filename)
    image.save(path)
    return os.path.abspath(path)
