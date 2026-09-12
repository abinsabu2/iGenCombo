import argparse
import sys

try:
    from src.gen import collect_ref_images, generate
except ImportError:
    from gen import collect_ref_images, generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image via SD Turbo")
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument("--model", default=None, help="HF model id (default: stabilityai/sd-turbo)")
    parser.add_argument("--out", default="out", help="output directory")
    parser.add_argument("--steps", type=int, default=1, help="inference steps")
    parser.add_argument("--guidance", type=float, default=0.0, help="guidance scale (0.0 for turbo)")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--height", type=int, default=512, help="image height")
    parser.add_argument("--width", type=int, default=512, help="image width")
    parser.add_argument("--source", default=None, help="directory of reference images")
    parser.add_argument("--ref-scale", type=float, default=0.8, help="IP-Adapter scale")
    args = parser.parse_args()
    ref_images = None
    if args.source is not None:
        try:
            ref_images = collect_ref_images(args.source)
        except FileNotFoundError as e:
            parser.error(str(e))
    result = generate(prompt=args.prompt, model=args.model, out_dir=args.out, steps=args.steps, guidance=args.guidance, seed=args.seed, height=args.height, width=args.width, ref_images=ref_images, ref_scale=args.ref_scale)
    print(result)


if __name__ == "__main__":
    main()
