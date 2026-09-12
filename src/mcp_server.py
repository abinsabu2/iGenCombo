import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

try:
    from src.gen import collect_ref_images, generate
except ImportError:
    from gen import collect_ref_images, generate

mcp = FastMCP("iGenCombo", host="127.0.0.1")


@mcp.tool()
def generate_image(prompt: str, steps: int = 1, seed: int | None = None, out_dir: str = "out", model: str | None = None, guidance: float = 0.0, height: int = 512, width: int = 512, source: str | None = None, ref_scale: float = 0.8) -> str:
    """Generate an image from a text prompt. Returns absolute path to PNG."""
    ref_images = None
    if source is not None:
        ref_images = collect_ref_images(source)
    return generate(prompt=prompt, steps=steps, seed=seed, out_dir=out_dir, model=model, guidance=guidance, height=height, width=width, ref_images=ref_images, ref_scale=ref_scale)


if __name__ == "__main__":
    mcp.run(transport="stdio")
