import os
import sys
import json
import argparse
import torch

# ================= Path setup =================
# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, CLIPTokenizer, GemmaTokenizer, CLIPTextModelWithProjection, AutoModelForCausalLM

# Import model definitions
from diffusion.configs import DiTConfig, FuseDiTConfig
from diffusion.models import DiT, FuseDiT, AdaFuseDiT
from diffusion.pipelines import DiTPipeline, FuseDiTPipeline, FuseDiTPipelineWithCLIP, AdaFuseDiTPipeline


def get_torch_dtype(dtype_str):
    dtype_map = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def clean_state_dict(state_dict):
    """
    Normalize weight keys:
    1. Remove 'module.' (from DDP/DeepSpeed)
    2. Remove '_orig_mod.' (from torch.compile)
    3. Remove 'shadow_params.' (from some EMA implementations)
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # Iteratively strip prefixes
        while new_k.startswith("module."):
            new_k = new_k[7:]
        while new_k.startswith("_orig_mod."):
            new_k = new_k[10:]
        new_state_dict[new_k] = v
    return new_state_dict


def load_generic_pt_file(file_path):
    """
    Generic loader for .pt files.
    Supports DeepSpeed checkpoints (mp_rank_00...) and EMA checkpoints (ema.pt).
    """
    print(f"Loading checkpoint directly from file: {file_path}")
    
    # map_location="cpu" prevents OOM on GPU
    checkpoint = torch.load(file_path, map_location="cpu")
    
    state_dict = None

    # .pt files are often dicts; find the key holding weights
    if isinstance(checkpoint, dict):
        # 1. Check common EMA key
        if "shadow_params" in checkpoint:
            print("Found 'shadow_params' (EMA) key in checkpoint...")
            state_dict = checkpoint["shadow_params"]
        # 2. Check common DeepSpeed/DDP key
        elif "module" in checkpoint:
            print("Found 'module' key in checkpoint...")
            state_dict = checkpoint["module"]
        # 3. Check standard state_dict
        elif "state_dict" in checkpoint:
            print("Found 'state_dict' key in checkpoint...")
            state_dict = checkpoint["state_dict"]
        # 4. If the dict is tensors only, treat it as state_dict
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
             print("Checkpoint is a raw state dict.")
             state_dict = checkpoint
        else:
            print("[Warning] Could not identify explicit key. Trying to use dict as is.")
            state_dict = checkpoint
            
    return state_dict


def load_checkpoint_weights(path, use_ema=False):
    """
    Generic loader.
    :param path: file path or directory path
    :param use_ema: when directory, prefer ema.pt
    """
    # 1. If user points to a file (e.g., .../ema.pt)
    if os.path.isfile(path):
        return load_generic_pt_file(path)
    
    # 2. If directory, search by priority
    print(f"Scanning checkpoint directory: {path}")
    
    # Candidate filenames
    if use_ema:
        # If EMA requested, try ema.pt first
        candidates = [
            "ema.pt",
            "ema_state_dict.pt",
            # fallback to regular weights if EMA not found
            "mp_rank_00_model_states.pt", 
            "model.safetensors", 
            "pytorch_model.bin",
        ]
        print("-> Mode: Prefer EMA weights")
    else:
        candidates = [
            "mp_rank_00_model_states.pt",
            "pytorch_model/mp_rank_00_model_states.pt",
            "model.safetensors", 
            "pytorch_model.bin",
            "ema.pt" # fallback candidate
        ]
    
    target_file = None
    for fname in candidates:
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            target_file = full_path
            print(f"Found match: {fname}")
            break
            
    if target_file:
        return load_checkpoint_weights(target_file, use_ema)
    
    raise FileNotFoundError(f"Could not find valid model file in {path}. Candidates checked: {candidates}")


def main(args):
    weight_dtype = get_torch_dtype(args.dtype)
    print(f"Target Dtype: {weight_dtype}")

    # ================= 1. Resolve config path =================
    config_path = args.checkpoint
    if os.path.isfile(config_path):
        # If pointing to a file (e.g., ema.pt), step back to checkpoint dir
        config_path = os.path.dirname(config_path) 
        
    # If config not in checkpoint root, look one level up
    if not os.path.exists(os.path.join(config_path, "config.json")):
        parent_dir = os.path.dirname(config_path)
        if os.path.exists(os.path.join(parent_dir, "config.json")):
            config_path = parent_dir
            
    print(f"Loading config from: {config_path}")
    
    try:
        if args.type == "dit":
            config = DiTConfig.from_pretrained(config_path)
            model_cls = DiT
        elif "fuse-dit" in args.type:
            config = FuseDiTConfig.from_pretrained(config_path)
            model_cls = FuseDiT
        elif args.type == "adafusedit":
            config = DiTConfig.from_pretrained(config_path)
            model_cls = AdaFuseDiT
        else:
            raise ValueError(f"Unknown type: {args.type}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # ================= 2. Initialize model =================
    print(f"Initializing model architecture: {model_cls.__name__}...")
    transformer = model_cls(config)

    # ================= 3. Load weights =================
    # Pass use_ema flag through
    raw_state_dict = load_checkpoint_weights(args.checkpoint, use_ema=args.use_ema)
    
    print("Cleaning state dict keys...")
    state_dict = clean_state_dict(raw_state_dict)
    
    print("Applying state dict to model...")
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"[Warn] Missing keys ({len(missing)}): {missing[:3]} ...")
    if len(unexpected) > 0:
        print(f"[Warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
        
    transformer.to(dtype=weight_dtype)

    # ================= 4. Handle LLM/Tokenizer =================
    llm_path = args.llm_path
    if llm_path is None:
        if hasattr(config, "base_config") and hasattr(config.base_config, "_name_or_path"):
            llm_path = config.base_config._name_or_path
        elif hasattr(config, "_name_or_path"):
            llm_path = config._name_or_path
            
    print(f"Loading auxiliary models from: {llm_path}")
    
    tokenizer = None
    lm = None

    if llm_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        except:
            print("AutoTokenizer failed, trying GemmaTokenizer...")
            tokenizer = GemmaTokenizer.from_pretrained(llm_path)
        
        # Try to load LLM
        try:
            from transformers import AutoModelForCausalLM
            temp_lm = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=weight_dtype)
            if hasattr(temp_lm, "model") and isinstance(temp_lm.model, torch.nn.Module):
                lm = temp_lm.model
            else:
                lm = temp_lm
        except Exception:
            lm = None

        # If that fails, try as VLM
        if lm is None:
            try:
                from transformers import AutoModelForImageTextToText
                vl = AutoModelForImageTextToText.from_pretrained(llm_path, torch_dtype=weight_dtype)
                for attr in ["language_model", "text_model", "model"]:
                    if hasattr(vl, attr):
                        lm_part = getattr(vl, attr)
                        if hasattr(lm_part, "model") and isinstance(lm_part.model, torch.nn.Module):
                            lm = lm_part.model
                        else:
                            lm = lm_part
                        break
            except Exception:
                pass
       
    if lm is not None:
        lm = lm.to(dtype=weight_dtype)

    # ================= 5. Build pipeline =================
    print("Building Pipeline...")
    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae")
        
        clip, clip_tok = None, None
        if args.type == "fuse-dit-clip":
            clip = CLIPTextModelWithProjection.from_pretrained(args.clip_l, subfolder="text_encoder")
            clip_tok = CLIPTokenizer.from_pretrained(args.clip_l, subfolder="tokenizer")

        pipeline = None
        if args.type == "dit":
            pipeline = DiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, llm=lm)
        elif args.type == "adafusedit":
            pipeline = AdaFuseDiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, llm=lm)
        elif args.type == "fuse-dit":
            pipeline = FuseDiTPipeline(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer)
        elif args.type == "fuse-dit-clip":
            pipeline = FuseDiTPipelineWithCLIP(transformer=transformer, scheduler=scheduler, vae=vae, tokenizer=tokenizer, clip=clip, clip_tokenizer=clip_tok)
            
    except Exception as e:
        print(f"Error building pipeline: {e}")
        return

    # ================= 6. Save =================
    
    # Auto-select folder name
    # Rule: if --use_ema is set or filename contains 'ema', append suffix
    is_ema_mode = False
    
    if args.use_ema:
        is_ema_mode = True
    elif os.path.isfile(args.checkpoint) and "ema" in os.path.basename(args.checkpoint).lower():
        is_ema_mode = True
        
    folder_name = "pipeline_ema" if is_ema_mode else "pipeline"
    
    # Determine base path
    if os.path.isfile(args.checkpoint):
        save_base = os.path.dirname(args.checkpoint)
    else:
        save_base = args.checkpoint

    output_dir = os.path.join(save_base, folder_name)
    
    print(f"Saving pipeline to: {output_dir}")
    pipeline.save_pretrained(output_dir)
    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accepts a directory or an explicit ema.pt path
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory or specific .pt file")
    
    # Option: prefer EMA weights
    parser.add_argument("--use_ema", action="store_true", help="If loading from a directory, prefer 'ema.pt', and save to 'pipeline_ema'")
    
    parser.add_argument("--type", type=str, default="fuse-dit", choices=["dit", "fuse-dit", "fuse-dit-clip", "adafusedit"])
    parser.add_argument("--llm_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    parser.add_argument("--scheduler", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--vae", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--clip_l", type=str, default="stabilityai/stable-diffusion-3-medium-diffusers")
    
    args = parser.parse_args()
    main(args)
