import argparse

from lightning import seed_everything

from diffusion.pipelines import FuseDiTPipeline
import torch


def main(args):
    seed_everything(args.seed)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {args.dtype}")

    pipeline = FuseDiTPipeline.from_pretrained(args.checkpoint, torch_dtype=torch_dtype).to("cuda")

    with torch.autocast("cuda", dtype=torch_dtype):
        image = pipeline(
            args.prompt,
            width=args.resolution,
            height=args.resolution,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_cache=True,
        )[0][0]
    image.save(args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/path/to/checkpoint")
    parser.add_argument("--prompt", type=str, default="The national flag of the country where Yellowstone National Park is located.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--save_path", type=str, default="test.jpg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()
    main(args)
