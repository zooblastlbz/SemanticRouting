
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from accelerate import PartialState
import fire
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import torch
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
    
    with open(opt.gen.metadata_file, "r") as f:
        metadata = json.load(f)
    prompts=[]
    for item in metadata.values():
        prompts.append(item)
    
    
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
    
    with distributed_state.split_between_processes(prompts) as samples:
        for model in tqdm(opt.pipeline.ckpt_path):
            output_dir=os.path.join(model, f"draw-{int(opt.gen.scale)}-{int(opt.gen.steps)}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            pipe = load_pipeline(opt.pipeline.model_type, os.path.join(model, "pipeline"), torch_dtype)
            for index in tqdm(range(len(samples))):
                
                sample = samples[index]
                prompt = sample['caption']
                generator = torch.manual_seed(opt.gen.seed)
                with torch.autocast("cuda", dtype=torch_dtype):
                    images = pipe(
                        prompt=prompt,
                        height=opt.gen.H,
                        width=opt.gen.W,
                        num_inference_steps=opt.gen.steps,
                        guidance_scale=opt.gen.scale,
                        num_images_per_prompt=opt.gen.n_samples,
                        negative_prompt=opt.gen.negative_prompt or None,
                        generator=generator,
                        instruction=opt.gen.instruction,
                    )[0]
                
                

                for i, image in enumerate(images):
                    image.save(os.path.join(output_dir, f"{index}_sample_{i}.png"))


def main(config_file):
    hparams = OmegaConf.load(config_file)
    generate(hparams)


if __name__ == "__main__":
    fire.Fire(main)