# iGenCombo

SD Turbo image generation via CLI, Python API, and MCP (opencode). Default model `stabilityai/sd-turbo` generates 512x512 in 1 step at guidance 0.0.

## Requirements

- Python 3.14.2 — installed GLOBALLY on this host, no venv in repo
- `pip` (via `python3 -m pip`)
- Dependencies pinned in `requirements.txt`:

```
torch==2.10.0
diffusers==0.37.1
transformers==5.1.0
accelerate==1.15.0
mcp==1.26.0
huggingface_hub==1.31.0
```

Weights auto-download to `~/.cache/huggingface/hub` on first use (Hugging Face hub cache). Minimal runnable set ~4-5 GB; ensure ~10 GB free for cache + outputs.

## Install

`install.sh` exists at repo root. It does:

```bash
python3 -m pip install -r requirements.txt
# CPU-only torch hint (commented, no CUDA):
# python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
```

Run it:

```bash
chmod +x install.sh
./install.sh
```

Or manually:

```bash
python3 -m pip install -r requirements.txt
```

Or individually:

```bash
python3 -m pip install torch==2.10.0 diffusers==0.37.1 transformers==5.1.0 accelerate==1.15.0 mcp==1.26.0 huggingface_hub==1.31.0
```

Device is picked automatically in `src/gen.py`: `cuda` -> `mps` -> `cpu`. `dtype` is `fp16` on `cuda`/`mps`, `fp32` on `cpu`. On Apple Silicon, `mps` is used when `torch.backends.mps.is_available()`.

## Hugging Face setup

`huggingface_hub==1.31.0` is a declared dependency (see `requirements.txt`). The `hf` CLI ships with it (`/Library/Frameworks/Python.framework/Versions/3.14/bin/hf` on this host).

Cache location: `~/.cache/huggingface/hub` by default. Relocate with either `HF_HOME` (base dir, e.g. `HF_HOME=/path/to/cache`) or `HF_HUB_CACHE` (direct hub dir, e.g. `HF_HUB_CACHE=/path/to/cache/hub`). `HF_HUB_CACHE` takes precedence; legacy `HUGGINGFACE_HUB_CACHE` also honored. Verified in `huggingface_hub/constants.py:158-182` for `1.31.0` (`HF_HOME` → `HF_HUB_CACHE` → `HUGGINGFACE_HUB_CACHE` fallback chain).

Measured on-disk sizes (`du -sh ~/.cache/huggingface/hub/models--*` on this host):

```
7.1G  models--stabilityai--sd-turbo
16G   models--stabilityai--sdxl-turbo  (includes ~3.4G incomplete blob; ~13G without)
0B    models--h94--IP-Adapter          (not yet downloaded; only lock files present)
```

`h94/IP-Adapter` weights (`sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors` + `models/image_encoder`) are fetched on first use with `--source` / `source` (IP-Adapter path).

Authentication: `hf auth login` (verified via `hf auth --help` on `1.31.0`; `huggingface-cli` is deprecated and prints `Use hf instead`). Example:

```bash
hf auth login
# or non-interactive: hf auth login --token <YOUR_TOKEN>
```

`stabilityai/sd-turbo`, `stabilityai/sdxl-turbo`, and `h94/IP-Adapter` are **public** — no token required. Login is only needed for gated or private models. Do not commit tokens; use placeholders only.

Offline use (verified: `huggingface_hub/constants.py:192` reads `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` and `utils/_http.py` raises when offline):

```bash
HF_HUB_OFFLINE=1 python3 -m src.cli --prompt "a cat" --seed 42
```

Requires weights already cached; otherwise the run will fail without network access.

Prefetch (optional, avoids first-run download latency):

```bash
hf download stabilityai/sd-turbo --include "model_index.json" --include "scheduler/**" --include "tokenizer*/**" --include "text_encoder/*.safetensors" --include "text_encoder/config.json" --include "unet/diffusion_pytorch_model.safetensors" --include "unet/config.json" --include "vae/diffusion_pytorch_model.safetensors" --include "vae/config.json"
```

Accelerated downloads: `hf_transfer` (Rust, `HF_HUB_ENABLE_HF_TRANSFER=1`) is deprecated in `1.31.0` (`constants.py:293-301` warns to use `HF_XET_HIGH_PERFORMANCE` instead). The replacement `hf-xet==1.6.0` is already installed transitively via `huggingface_hub`. To enable it:

```bash
HF_XET_HIGH_PERFORMANCE=1 python3 -m src.cli --prompt "a cat" --seed 42
```

Optional legacy path (not recommended): `python3 -m pip install hf_transfer` + `HF_HUB_ENABLE_HF_TRANSFER=1` — will emit a `FutureWarning` on `1.31.0`.

## Quickstart

One copy-paste command that works:

```bash
python3 -m src.cli --prompt "a serene mountain landscape at sunset" --out out --steps 1 --seed 42
```

Prints an absolute path to the PNG, e.g. `/Users/abinsabu/Documents/iGenCombo/out/20260912_172028_42.png`.

Alternative invocation:

```bash
python3 src/cli.py --prompt "a serene mountain landscape at sunset" --out out --steps 1 --seed 42
```

## CLI Reference

Flags defined in `src/cli.py:11-21`:

| Flag | Type | Default | Required | Description |
|------|------|---------|----------|-------------|
| `--prompt` | str | — | Yes | Text prompt |
| `--model` | str | `None` (= `stabilityai/sd-turbo`) | No | Hugging Face model ID (`AutoPipelineForText2Image`-compatible) |
| `--out` | str | `"out"` | No | Output directory |
| `--steps` | int | `1` | No | Inference steps (1-4 recommended for turbo) |
| `--guidance` | float | `0.0` | No | Guidance scale (0.0 for turbo) |
| `--seed` | int | `None` (random) | No | Random seed for reproducibility |
| `--height` | int | `512` | No | Image height in pixels |
| `--width` | int | `512` | No | Image width in pixels |
| `--source` | str | `None` | No | Directory of reference images (IP-Adapter, see below) |
| `--ref-scale` | float | `0.8` | No | IP-Adapter scale |

Notes:

- `--ref-scale` is hyphenated on the CLI but `ref_scale` in Python/MCP.
- Any Hugging Face `AutoPipelineForText2Image`-compatible ID works for `--model`.
- `out_dir` default is `"out"`, resolved relative to REPO ROOT when not absolute (`src/gen.py:80-81`). That is, `--out out` writes to `<repo>/out` regardless of current working directory. Absolute paths are used as-is. This surprises people — use an absolute path if you want cwd-relative behavior.

Examples:

```bash
# SD Turbo default (512x512, 1 step)
python3 -m src.cli --prompt "a cat wearing a spacesuit" --seed 0

# Override model
python3 -m src.cli --prompt "a cat" --model stabilityai/sdxl-turbo --steps 2 --height 1024 --width 1024

# Custom output directory (absolute)
python3 -m src.cli --prompt "a cat" --out /tmp/my-images --seed 123

# With reference images (auto-switches to SDXL, see Reference Images)
python3 -m src.cli --prompt "a portrait photo" --source /path/to/refs --ref-scale 0.8 --seed 42
```

## Python API

`generate()` at `src/gen.py:32`:

```python
generate(prompt, steps=1, seed=None, out_dir="out", model=None, guidance=0.0, height=512, width=512, *, ref_images=None, ref_scale=0.8) -> str
```

Full signature:

```python
def generate(
    prompt: str,
    steps: int = 1,
    seed: int | None = None,
    out_dir: str = "out",
    model: str | None = None,
    guidance: float = 0.0,
    height: int = 512,
    width: int = 512,
    *,
    ref_images: list[str] | None = None,
    ref_scale: float = 0.8,
) -> str:
```

- Returns absolute path to the PNG.
- `model=None` means `stabilityai/sd-turbo`.
- `seed=None` picks a random seed (`random.randint(0, 2**32 - 1)`).
- Output filename: `{YYYYMMDD_HHMMSS}_{seed}.png` (`src/gen.py:102-104`), e.g. `20260912_172056_12345.png`. Seed is embedded in the filename.
- `out_dir` default `"out"`, resolved relative to repo root when not absolute (`src/gen.py:80-81`).
- `ref_images` is a list of file paths (or `None`). Use `collect_ref_images()` to build it from a directory, or pass paths directly.
- `ref_scale` is the IP-Adapter scale (default `0.8`).
- Device/dtype handled internally: `cuda` -> `mps` -> `cpu`; `fp16` on `cuda`/`mps`, `fp32` on `cpu`.

Example:

```python
from src.gen import generate, collect_ref_images

# Basic
path = generate("a serene mountain landscape at sunset", seed=42)
print(path)  # /.../out/20260912_172028_42.png

# With reference images
refs = collect_ref_images("/path/to/refs")
path = generate("a portrait photo", seed=42, ref_images=refs, ref_scale=0.8)
print(path)

# Custom size and model
path = generate("a cat", model="stabilityai/sdxl-turbo", height=1024, width=1024, steps=2, seed=0)
print(path)
```

`collect_ref_images()` at `src/gen.py:12`:

```python
def collect_ref_images(source: str) -> list[str]:
```

- `source` must be a directory (raises `FileNotFoundError` if not a dir).
- Non-recursive `iterdir()`, extensions `.jpg` `.jpeg` `.png`, case-insensitive, sorted.
- Raises `FileNotFoundError` if no images found.

## MCP Setup + Usage

Server defined in `src/mcp_server.py:13`:

```python
mcp = FastMCP("iGenCombo", host="127.0.0.1")
```

Transport is `stdio` (`src/mcp_server.py:26`: `mcp.run(transport="stdio")`).

Tool `generate_image` has the same params as `generate()` but with `source: str | None` instead of `ref_images`:

```python
@mcp.tool()
def generate_image(prompt: str, steps: int = 1, seed: int | None = None, out_dir: str = "out", model: str | None = None, guidance: float = 0.0, height: int = 512, width: int = 512, source: str | None = None, ref_scale: float = 0.8) -> str:
```

`source` is a directory path (same rules as `--source`); `None` means no reference images.

### opencode config

Add this block to your opencode config:

```json
"mcp": { "igencombo": { "type": "local", "command": ["python3", "-m", "src.mcp_server"], "cwd": "/Users/abinsabu/Documents/iGenCombo", "enabled": true, "timeout": 900000 } }
```

`cwd` must be an absolute path to your checkout. `timeout` is 900000 ms (15 minutes) to allow model load + generation.

opencode must be restarted for config or code changes to take effect.

Verify the server starts:

```bash
python3 -m src.mcp_server
# should start and wait on stdio (no error)
```

## Reference Images / IP-Adapter

`--source` / `source` accepts a DIRECTORY ONLY, not a single file (`src/gen.py:12-17` raises `FileNotFoundError` if not a dir). Scanning is non-recursive `iterdir()`, extensions `.jpg` `.jpeg` `.png`, case-insensitive, sorted. Empty or non-image directories raise `FileNotFoundError`.

IP-Adapter details:

- Adapter: `h94/IP-Adapter` + `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors` — SDXL-ONLY.
- It is a **face adapter** (`plus-face`), tuned for faces. Pointing it at landscapes, objects, or non-face subjects will produce poor or confusing results.

Behavior (`src/gen.py:51-62`):

| Condition | Result |
|-----------|--------|
| Reference images + default model (`model=None`) | Auto-switches to `stabilityai/sdxl-turbo` and bumps 512x512 to 1024x1024, printing a notice to stderr |
| Reference images + explicit non-SDXL model (e.g. `stabilityai/sd-turbo`) | Raises `ValueError: IP-Adapter reference images require an SDXL model` |
| Reference images + explicit SDXL model (e.g. `stabilityai/sdxl-turbo`) | Uses the given model as-is |
| No reference images | No switching, uses `stabilityai/sd-turbo` at 512x512 by default |

```bash
# Auto-switches to sdxl-turbo @ 1024x1024 (default model + refs)
python3 -m src.cli --prompt "a portrait photo" --source /path/to/face-refs --seed 42

# Explicit SDXL model — no auto-switch needed
python3 -m src.cli --prompt "a portrait photo" --source /path/to/face-refs --model stabilityai/sdxl-turbo --seed 42

# Explicit non-SDXL + refs — fails with ValueError
python3 -m src.cli --prompt "a portrait photo" --source /path/to/face-refs --model stabilityai/sd-turbo
# ValueError: IP-Adapter reference images require an SDXL model, got 'stabilityai/sd-turbo'
```

Reference images are validated before weights load, so a bad path never triggers adapter download. Each image is verified via `PIL.Image.verify()` and `convert("RGB")`; unreadable files raise `ValueError`.

## Performance

Measured on M2 Pro. No embellishment.

- Pipeline is cached per process via `lru_cache(maxsize=2)` keyed on `(model, device, dtype, with_ip_adapter)` (`src/gen.py:24`).
- First call in a process ~12s (model load). Subsequent calls ~0.3s at 256x256 / ~1.0s at 512x512.
- `sdxl-turbo` @ 1024x1024 2 steps ~36s for comparison.
- Cold first-ever run also downloads weights (~13s extra on first disk hit, then cached to `~/.cache/huggingface/hub`).
- Consequence: long-lived MCP server is fast after the first request; each fresh `cli.py` invocation pays the load again.

```
Call 1 (cold, no cache)          ~12s  (+ ~13s download on first-ever run)
Call 2+ @ 256x256                ~0.3s
Call 2+ @ 512x512                ~1.0s
sdxl-turbo @ 1024x1024, 2 steps   ~36s
```

## Seeding / Reproducibility

- `--seed` / `seed` gives reproducible output: same seed + same params (prompt, model, steps, guidance, height, width, ref images, ref_scale) = pixel-identical output (verified).
- Omit `--seed` for random. A random seed is chosen via `random.randint(0, 2**32 - 1)` and printed to stderr.
- Seed is embedded in the output filename: `{YYYYMMDD_HHMMSS}_{seed}.png`.
- Torch seeding: `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` on cuda + `torch.Generator(device=...).manual_seed(seed)` passed to the pipeline.

```bash
# Reproducible — same seed, same output
python3 -m src.cli --prompt "a cat" --seed 42 --out /tmp/a
python3 -m src.cli --prompt "a cat" --seed 42 --out /tmp/b
# /tmp/a/..._42.png and /tmp/b/..._42.png are pixel-identical

# Random — different output each run
python3 -m src.cli --prompt "a cat"
```

## Tests

```bash
# Fast, no weights, no GPU — validates defaults, model switching, source handling
python3 test_defaults.py

# Full suite (20 tests)
python3 -m pytest tests/ -v
# or
pytest tests/
```

## Troubleshooting

- **CPU slow** — lower steps: `--steps 1` or `--steps 2` for quick previews; CPU inference is much slower than GPU. Expect ~1s at 512x512 after warm load, but first load is ~12s.
- **MPS (Apple Silicon)** — `src/gen.py` auto-selects `mps` when `torch.backends.mps.is_available()`; ensure PyTorch with MPS support (`torch==2.10.0` includes it). If MPS unavailable it falls back to `cpu`. `dtype` is `fp16` on `mps`, `fp32` on `cpu`.
- **Cache dir** — Hugging Face models cache to `~/.cache/huggingface/hub` by default. Override with `HF_HOME` or `HF_HUB_CACHE`:
  ```bash
  HF_HOME=/path/to/cache python3 -m src.cli --prompt "test"
  ```
  Clear or relocate cache if disk is full.
- **Restart opencode after config/code changes** — opencode reads MCP config only on launch. After editing the `mcp` block or changing `src/`, fully quit and reopen opencode.
- **`--source` must be a directory** — passing a single file path raises `FileNotFoundError: source directory not found`. Pass the containing directory instead.
- **`ValueError: IP-Adapter reference images require an SDXL model`** — you passed `--source` with an explicit non-SDXL `--model`. Either omit `--model` (auto-switches to `sdxl-turbo`) or pass an SDXL model explicitly.
- **`FileNotFoundError: no images found in source directory`** — directory exists but contains no `.jpg`/`.jpeg`/`.png` files (case-insensitive, non-recursive). Check extensions and that files are directly inside the directory, not in subdirectories.
- **`unreadable reference image`** — file has an image extension but is corrupt or not a valid image. Re-export the file.

## Known Issues / Housekeeping

- Repo has NO `.gitignore` and `out/` is committed (3 PNGs tracked). Stray empty `source/` and `output/` dirs and `.DS_Store` files are also present/tracked.
- Recommended `.gitignore` (not yet created — no files deleted):
  ```
  out/
  __pycache__/
  .DS_Store
  .pytest_cache/
  source/
  output/
  ```
  Do not commit generated images or cache artifacts. `out/` is the default output directory and should be ignored.
- Generated images carry NO metadata — no model, prompt, or seed is embedded in the PNG. Filename contains timestamp + seed only. Track params externally if you need reproducibility records.
- `out_dir` resolved relative to repo root (not cwd) when not absolute — can surprise callers running from a different directory.

## Model

Default: `stabilityai/sd-turbo` (SD Turbo, 512-native, 1-step). Override via `--model` / `model` param. Any Hugging Face `AutoPipelineForText2Image`-compatible ID works. Reference images (IP-Adapter) require an SDXL model; when `source` is given with the default model the pipeline auto-switches to `stabilityai/sdxl-turbo` at 1024x1024.
