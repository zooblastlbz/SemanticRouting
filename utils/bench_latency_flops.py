import argparse
import time

import torch

from diffusion.pipelines import AdaFuseDiTPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Measure latency and FLOPs for AdaFuseDiT pipeline.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pipeline checkpoint (from_pretrained).")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Compute dtype.")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution (height=width).")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.0, help="CFG guidance scale.")
    parser.add_argument("--prompt", type=str, default="a colorful mural on a brick wall", help="Prompt text.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before measuring latency.")
    parser.add_argument("--repeat", type=int, default=3, help="Number of measured runs for averaging latency.")
    return parser.parse_args()


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def count_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def count_fusion_params(transformer):
    total = 0
    if hasattr(transformer, "text_fusion_module") and transformer.text_fusion_module is not None:
        total += count_params(transformer.text_fusion_module)
    if hasattr(transformer, "text_fusion_modules") and transformer.text_fusion_modules is not None:
        total += count_params(transformer.text_fusion_modules)
    if hasattr(transformer, "text_fusion_weight") and transformer.text_fusion_weight is not None:
        total += transformer.text_fusion_weight.numel()
    if hasattr(transformer, "text_fusion_weights") and transformer.text_fusion_weights is not None:
        total += sum(w.numel() for w in transformer.text_fusion_weights)
    return total


@torch.inference_mode()
def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch_dtype = get_dtype(args.dtype)

    pipeline = AdaFuseDiTPipeline.from_pretrained(args.checkpoint, torch_dtype=torch_dtype).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Parameter counts
    total_params = count_params(pipeline)
    fusion_params = count_fusion_params(pipeline.transformer) if hasattr(pipeline, "transformer") else 0
    llm_params = count_params(pipeline.transformer.llm) if hasattr(pipeline.transformer, "llm") else 0
    dit_params = count_params(pipeline.transformer.dit) if hasattr(pipeline.transformer, "dit") else count_params(pipeline.transformer)
    non_llm_params = total_params - llm_params

    def run_once():
        return pipeline(
            prompt=args.prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            output_type="latent",  # decode skipped to keep focus on transformer
        )

    # Warmup
    for _ in range(args.warmup):
        run_once()

    # Latency
    times = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - start)
    avg_latency = sum(times) / len(times)

    # FLOPs (single run)
    flops_total = None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            run_once()
        flops_total = sum([e.flops for e in prof.key_averages() if e.flops is not None])
    except Exception as exc:
        print(f"[warn] FLOPs profiling failed: {exc}")

    print(f"Device: {device}, dtype: {torch_dtype}")
    print(f"Resolution: {args.resolution}x{args.resolution}, steps: {args.steps}, guidance: {args.guidance}")
    print(f"Total params (incl. llm): {total_params/1e6:.2f} M")
    print(f"Params excluding llm: {non_llm_params/1e6:.2f} M")
    print(f"Transformer params (dit part): {dit_params/1e6:.2f} M")
    if fusion_params:
        print(f"Fusion module params: {fusion_params/1e6:.2f} M")
    else:
        print("Fusion module params: 0.00 M")
    if llm_params and isinstance(pipeline, AdaFuseDiTPipeline):
        print(f"LLM params: {llm_params/1e6:.2f} M")
    print(f"Latency (average over {args.repeat} runs): {avg_latency*1000:.2f} ms")
    if flops_total is not None:
        print(f"Approx FLOPs (single run): {flops_total/1e12:.3f} TFLOPs")
    else:
        print("Approx FLOPs: unavailable")


if __name__ == "__main__":
    main()
