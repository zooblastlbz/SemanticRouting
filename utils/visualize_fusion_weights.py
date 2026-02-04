import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add current directory to sys.path to allow importing diffusion
sys.path.append(os.getcwd())

from diffusion.models import build_model
from diffusion.pipelines import AdaFuseDiTPipeline


def _parse_layer_list(layer_str):
    """Parse a comma-separated layer list like '0,3,5' into a set of ints."""
    if not layer_str:
        return None
    layers = []
    for part in layer_str.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            layers.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid layer id '{part}' in --layers, must be integers separated by commas")
    return set(layers)


def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = None
    
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    elif os.path.isdir(checkpoint_path):
        # Try to find mp_rank_00_model_states.pt (common in DeepSpeed)
        p = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        if os.path.exists(p):
            print(f"Found model states at {p}")
            payload = torch.load(p, map_location="cpu")
            if "module" in payload:
                state_dict = payload["module"]
            else:
                state_dict = payload
        else:
            # Try DeepSpeed ZeRO utility
            print("Attempting to load using DeepSpeed ZeRO utility...")
            try:
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
            except ImportError:
                print("DeepSpeed not installed or failed to import. Cannot load ZeRO checkpoint.")
            except Exception as e:
                print(f"Failed to load ZeRO checkpoint: {e}")
                
    if state_dict is None:
        raise ValueError(f"Could not load state dict from {checkpoint_path}")

    # Handle 'module.' prefix if present (DDP/DeepSpeed)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")


def load_from_pipeline(checkpoint_path):
    """Load model from a saved pipeline directory."""
    print(f"Loading pipeline from {checkpoint_path}...")
    pipeline = AdaFuseDiTPipeline.from_pretrained(checkpoint_path)
    model = pipeline.transformer
    print("Pipeline loaded successfully.")
    return model


def is_pipeline_dir(checkpoint_path):
    """Check if the path is a Hugging Face pipeline directory."""
    if not os.path.isdir(checkpoint_path):
        return False
    # Check for typical Hugging Face pipeline files
    indicators = ["model_index.json", "config.json", "pytorch_model.bin", "model.safetensors"]
    return any(os.path.exists(os.path.join(checkpoint_path, f)) for f in indicators)


def visualize(args):
    # Load config
    print(f"Loading config from {args.config}...")
    config = OmegaConf.load(args.config)
    
    # Mock HParams
    class HParams:
        def __init__(self, config):
            self.model = config.model if "model" in config else config
            self.trainer = argparse.Namespace(train_dit=False, train_llm=False)
            
    hparams = HParams(config)
    
    # Build or load model
    if is_pipeline_dir(args.checkpoint):
        print("Detected pipeline directory, loading from pipeline...")
        model = load_from_pipeline(args.checkpoint)
    else:
        print("Building model from config and loading checkpoint...")
        model = build_model(hparams)
        load_checkpoint(model, args.checkpoint)
    
    model.eval()
    
    # Prepare timesteps
    # Generate timesteps from 0 to 1000
    timesteps = torch.linspace(0, 1000, args.num_timesteps).long()
    # AdaFuseDiT uses normalized timesteps [0, 1] for the fusion module
    t_input = timesteps.float() / 1000.0
    t_input = t_input.to(model.device)  # Model might be on CPU, which is fine
    
    results = {}  # key -> weights (T, NumTextLayers)
    
    print("Extracting fusion weights...")
    if hasattr(model, "text_fusion_modules") and model.text_fusion_modules is not None:
        print("Detected Layer-wise Adaptive Fusion")
        for i, module in enumerate(model.text_fusion_modules):
            with torch.no_grad():
                # Use module forward to get weights; weights only depend on timestep embedding.
                dummy_hidden = [
                    torch.zeros(1, 1, module.feature_dim, device=t_input.device, dtype=t_input.dtype)
                    for _ in range(module.num_layers)
                ]
                _, w = module(dummy_hidden, t_input)  # (T, num_text_layers)
                results[f"Layer_{i}"] = w.cpu().numpy()
                
    elif hasattr(model, "text_fusion_module") and model.text_fusion_module is not None:
        print("Detected Global Adaptive Fusion")
        module = model.text_fusion_module
        with torch.no_grad():
            dummy_hidden = [
                torch.zeros(1, 1, module.feature_dim, device=t_input.device, dtype=t_input.dtype)
                for _ in range(module.num_layers)
            ]
            _, w = module(dummy_hidden, t_input)
            results["Global"] = w.cpu().numpy()
            
    elif hasattr(model, "text_fusion_weights") and model.text_fusion_weights is not None:
        print("Detected Layer-wise Static Fusion")
        for i, param in enumerate(model.text_fusion_weights):
            w = torch.softmax(param, dim=0).unsqueeze(0).repeat(len(timesteps), 1)
            results[f"Layer_{i}"] = w.detach().cpu().numpy()
            
    elif hasattr(model, "text_fusion_weight") and model.text_fusion_weight is not None:
        print("Detected Global Static Fusion")
        w = torch.softmax(model.text_fusion_weight, dim=0).unsqueeze(0).repeat(len(timesteps), 1)
        results["Global"] = w.detach().cpu().numpy()
        
    else:
        print("No fusion modules found in model. Is this AdaFuseDiT?")
        return

    # Filter for requested DiT layers when layer-wise fusion is present
    requested_layers = _parse_layer_list(args.layers)
    if requested_layers is not None:
        filtered = {}
        for key, weights in results.items():
            if key.startswith("Layer_"):
                try:
                    idx = int(key.split("_")[1])
                except Exception:
                    continue
                if idx not in requested_layers:
                    continue
            filtered[key] = weights
        if not filtered:
            print(f"No matching layers found for --layers={args.layers}. Available: {list(results.keys())}")
            return
        results = filtered
        print(f"Plotting only layers: {sorted(list(requested_layers))}")

    # Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use a colormap
    cm = plt.get_cmap("viridis")
    
    # Save CSVs for reference
    for key, weights in results.items():
        num_text_layers = weights.shape[1]
        csv_filename = f"fusion_weights_{key}.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        header = "Timestep," + ",".join([f"TextLayer_{i}" for i in range(num_text_layers)])
        t_np = timesteps.cpu().numpy()
        data_to_save = np.column_stack((t_np, weights))
        np.savetxt(csv_path, data_to_save, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f"Saved weights to {csv_path}")

    # Combined figure with one subplot per key/layer
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300})

    num_plots = len(results)
    cols = min(4, num_plots) if num_plots > 1 else 1
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, (key, weights) in enumerate(results.items()):
        ax = axes[ax_idx]
        num_text_layers = weights.shape[1]
        x = np.arange(num_text_layers)
        for t_idx, t_val in enumerate(timesteps):
            color = cm(t_idx / (len(timesteps) - 1) if len(timesteps) > 1 else 0.5)
            ax.plot(x, weights[t_idx], label=f"t={int(t_val)}", color=color, marker="o", markersize=3)
        ax.set_xlabel("Text Encoder Layer Index")
        ax.set_ylabel("Fusion Weight (Softmax)")
        ax.set_title(f"{key}")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    combined_path = os.path.join(args.output_dir, "fusion_weights_layers.pdf")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {combined_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize AdaFuseDiT fusion weights")
    parser.add_argument("--checkpoint", type=str, default="/path/to/checkpoint", help="Path to the checkpoint (file or folder)")
    parser.add_argument("--config", type=str, default="/path/to/config.yaml", help="Path to the model config file (.yaml)")
    parser.add_argument("--output_dir", type=str, default="visual/layerwise_fusion_plots", help="Directory to save plots")
    parser.add_argument("--num_timesteps", type=int, default=10, help="Number of timestep curves to plot")
    parser.add_argument("--layers", type=str, default="0,8,17,26,35", help="Comma-separated DiT layer ids to plot (only for layer-wise fusion). If omitted, all layers are plotted.")
    
    args = parser.parse_args()
    visualize(args)
