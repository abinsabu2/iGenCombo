# iGenCombo

SD Turbo image generation via CLI and MCP (Claude).

## Requirements

- Python 3.10+
- `pip`

## Install

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install torch==2.10.0 diffusers==0.37.1 transformers==5.1.0 accelerate==1.15.0 mcp==1.26.0
```

Quick install script:

```bash
chmod +x install.sh
./install.sh
```

## Model

Default: `stabilityai/sd-turbo` (SD Turbo, 512-native, 1-step).

Override via CLI flag:

```bash
python -m src.cli --prompt "a cat" --model stabilityai/sdxl-turbo
```

Any Hugging Face AutoPipelineForText2Image-compatible ID works. Reference images (IP-Adapter) require an SDXL model; when `source` is given with the default model the pipeline auto-switches to `stabilityai/sdxl-turbo` at 1024.

### Disk note

First run downloads SD Turbo weights (cached by `diffusers`/`transformers` in the Hugging Face cache). Minimal runnable set is ~4–5 GB. Ensure ~10 GB free for cache + outputs.

Prefetch only pipeline files:

```bash
hf download stabilityai/sd-turbo --include "model_index.json" --include "scheduler/**" --include "tokenizer*/**" --include "text_encoder/*.safetensors" --include "text_encoder/config.json" --include "unet/diffusion_pytorch_model.safetensors" --include "unet/config.json" --include "vae/diffusion_pytorch_model.safetensors" --include "vae/config.json"
```

## Run — CLI

```bash
python -m src.cli --prompt "a serene mountain landscape at sunset" --out out --steps 1 --seed 42
# or
python src/cli.py --prompt "a serene mountain landscape at sunset" --out out --steps 1 --seed 42
```

Output: absolute path to PNG in `out/` (e.g. `out/20260101_120000_42.png`).

Options:

- `--prompt` (required) — text prompt
- `--model` — HF model ID (default `stabilityai/sd-turbo`)
- `--out` — output directory (default `out`)
- `--steps` — inference steps (default `1`, 1–4 recommended for turbo)
- `--guidance` — guidance scale (default `0.0` for turbo)
- `--seed` — random seed (default random)

## MCP — Claude Config

Server runs over `stdio` and binds host `127.0.0.1` (see `src/mcp_server.py`).

Add to Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "igencombo": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/Users/abinsabu/Documents/iGenCombo"
    }
  }
}
```

Notes:

- `transport` is `stdio`; `host` is `127.0.0.1` (enforced in `src/mcp_server.py` via `FastMCP("iGenCombo", host="127.0.0.1")` + `mcp.run(transport="stdio")`).
- Use absolute `cwd` to your checkout. On macOS the config file is typically `~/Library/Application Support/Claude/claude_desktop_config.json`.

Verify:

```bash
python -m src.mcp_server
# should start and wait on stdio (no error)
```

## Troubleshooting

- **CPU slow** → lower steps: `--steps 1` or `--steps 2` for quick previews; CPU inference is much slower than GPU.
- **MPS (Apple Silicon)** → `src/gen.py` auto-selects `mps` when `torch.backends.mps.is_available()`; ensure PyTorch with MPS support (`torch==2.10.0` includes it). If MPS unavailable it falls back to `cpu`.
- **Cache dir** → Hugging Face models cache to `~/.cache/huggingface/hub` by default. Override with `HF_HOME` or `HF_HUB_CACHE`:
  ```bash
  HF_HOME=/path/to/cache python -m src.cli --prompt "test"
  ```
  Clear or relocate cache if disk is full.
- **Restart Claude after config** → Claude Desktop reads MCP config only on launch. After editing `claude_desktop_config.json`, fully quit and reopen Claude Desktop.
