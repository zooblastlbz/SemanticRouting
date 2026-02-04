"""
Adapted from:
https://github.com/djghosh13/geneval/blob/main/generation/diffusers_generate.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accelerate import PartialState
from tqdm import tqdm
import torch
from omegaconf import OmegaConf
import fire
import json

from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP, AdaFuseDiTPipeline


def load_pipeline(model_type: str, ckpt_path: str, dtype):
    if model_type == "dit":
        pipeline = DiTPipeline.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    elif model_type == "fuse-dit":
        pipeline = FuseDiTPipeline.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    elif model_type == "fuse-dit-clip":
        pipeline = FuseDiTPipelineWithCLIP.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    elif model_type == "adafusedit":
        pipeline = AdaFuseDiTPipeline.from_pretrained(ckpt_path, torch_dtype=dtype).to("cuda")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


@torch.no_grad()
def generate(opt):
    with open(opt.gen.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]
    
    dtype_str = opt.pipeline.get("dtype", "bf16")
    if dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    elif dtype_str == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    distributed_state = PartialState()
    with distributed_state.split_between_processes(list(enumerate(metadatas))) as samples:
        for model in tqdm(opt.pipeline.ckpt_path):
            if opt.pipeline.use_ema:
                pipe= load_pipeline(opt.pipeline.model_type, os.path.join(model, "pipeline_ema"), torch_dtype)
            else:
                pipe = load_pipeline(opt.pipeline.model_type, os.path.join(model, "pipeline"), torch_dtype)
            for index, metadata in tqdm(samples):
                if opt.pipeline.use_ema:
                    outpath = os.path.join(model,  f"geneval-ema-{int(opt.gen.scale)}", f"{index:0>5}")
                else:
                    outpath = os.path.join(model, f"geneval-{int(opt.gen.scale)}-{int(opt.gen.steps)}", f"{index:0>5}")
                os.makedirs(outpath, exist_ok=True)

                prompt = metadata['prompt']
                batch_size = opt.gen.batch_size
                print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

                sample_path = os.path.join(outpath, "samples")
                os.makedirs(sample_path, exist_ok=True)
                with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                    json.dump(metadata, fp)

                sample_count = 0
                for _ in range((opt.gen.n_samples + batch_size - 1) // batch_size):
                    # Generate images
                    generator = torch.manual_seed(opt.gen.seed)
                    with torch.autocast("cuda", dtype=torch_dtype):
                        images = pipe(
                            prompt=prompt,
                            height=opt.gen.H,
                            width=opt.gen.W,
                            num_inference_steps=opt.gen.steps,
                            guidance_scale=opt.gen.scale,
                            num_images_per_prompt=min(batch_size, opt.gen.n_samples - sample_count),
                            negative_prompt=opt.gen.negative_prompt or None,
                            generator=generator,
                            instruction=opt.gen.instruction,
                        )[0]
                    for image in images:
                        image.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                        sample_count += 1


def main(config_file):
    hparams = OmegaConf.load(config_file)
    generate(hparams)


if __name__ == "__main__":
    fire.Fire(main)