from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
import math
import os
import random
import shutil
import time
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import is_wandb_available
import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import CLIPTextModelWithProjection

from .data import get_dataloader
from .models import build_model, get_llm, update_self_attention_mask

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


class ProfileTimer:
    """Simple timing context manager."""
    def __init__(self, name, stats_dict=None, enabled=True):
        self.name = name
        self.stats_dict = stats_dict
        self.enabled = enabled
        self.start = 0

    def __enter__(self):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.time()
        return self

    def __exit__(self, *args):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - self.start
            if self.stats_dict is not None:
                self.stats_dict[self.name] = elapsed


def seed_everything(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer(ABC):
    """Base trainer."""
    
    def before_training(self):
        """Initialization before training."""
        if self.accelerator.is_main_process:
            if is_wandb_available():
                wandb.init(
                    project=self.hparams.trainer.project,
                    name=self.hparams.trainer.run,
                    config=OmegaConf.to_container(self.hparams, resolve=True)
                )
                # Log parameter sizes for later comparison
                if hasattr(self, "_param_stats"):
                    wandb.log(self._param_stats, step=self.global_step)
            print("***** Running training *****")
            print(f"  Total train batch size = {self.total_batch_size}")
            print(f"  Total optimization steps = {self.hparams.trainer.max_steps}")
            if hasattr(self, "_param_stats"):
                print(
                    f"  Parameters: total={self._param_stats['params/total']:,} "
                    f"({self._param_stats['params/total_m']:.2f}M), "
                    f"trainable={self._param_stats['params/trainable']:,} "
                    f"({self._param_stats['params/trainable_m']:.2f}M)"
                )
            self.progress_bar = tqdm(total=self.hparams.trainer.max_steps, initial=self.global_step)

    @abstractmethod
    def backward(self, loss):
        pass

    @abstractmethod
    def optimizer_step(self):
        pass

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(dtype=dtype)
        step_indices = [(self.noise_scheduler.timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def dit_training_step(self, batch):
        """DiT training step."""
        pixel_values = batch["pixel_values"]

        if self.hparams.model.encoder_type == "llm":
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(self.accelerator.device)

            #llm_attention_mask = update_self_attention_mask(
            #    attention_mask, 0, False, self.accelerator.device, 
            #    self.weight_dtype
            #)

            position_ids = torch.arange(input_ids.shape[1], device=self.accelerator.device).unsqueeze(0)

            with torch.no_grad():
                llm_output = self.llm(
                    input_ids.to(self.accelerator.device),
                    attention_mask.to(self.accelerator.device),
                    position_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=False,
                )
            all_hidden_states = llm_output[1]

            text_hidden_states_num = getattr(self.hparams.model, "text_hidden_states_num", 1)
            if text_hidden_states_num > 1:
                text_hidden_states = [
                    all_hidden_states[-text_hidden_states_num + i].to(dtype=self.weight_dtype)
                    for i in range(text_hidden_states_num)
                ]
            else:
                text_hidden_states_index = getattr(self.hparams.model, "text_hidden_states_index", -1)
                text_hidden_states = all_hidden_states[text_hidden_states_index].to(dtype=self.weight_dtype)
        else:
            raise ValueError(f"Unknown encoder type: {self.hparams.model.encoder}")

        model_input = pixel_values.to(self.accelerator.device)
        with torch.no_grad():
            model_input = self.vae.encode(model_input.float()).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        #model_input = model_input.to(dtype=self.weight_dtype)

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=torch.float32).to(self.accelerator.device)
        noisy_model_input = (1.0 - sigmas) * model_input.float() + sigmas * noise.float()
        noisy_model_input = noisy_model_input.to(dtype=self.weight_dtype)
        
        output = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.accelerator.device),
            text_hidden_states=text_hidden_states,
            attention_mask=attention_mask,
        )
        model_pred = output

        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas.to(dtype=model_pred.dtype)) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise.float() - model_input.float()

        loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()

        return loss

    def fusedit_training_step(self, batch):
        """FuseDiT training step."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if self.hparams.model.encoder_type == "clip-llm":
            clip_input_ids = batch["clip_input_ids"]
            with torch.no_grad():
                text_modulation_embeds = self.clip(clip_input_ids.to(self.accelerator.device)).text_embeds
            text_modulation_embeds = text_modulation_embeds.to(dtype=self.weight_dtype)
        else:
            text_modulation_embeds = None

        model_input = pixel_values.to(self.accelerator.device)
        with torch.no_grad():
            model_input = self.vae.encode(model_input.float()).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        #model_input = model_input.to(dtype=self.weight_dtype)

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=torch.float32).to(self.accelerator.device)
        noisy_model_input = (1.0 - sigmas) * model_input.float() + sigmas * noise.float()
        noisy_model_input = noisy_model_input.to(dtype=self.weight_dtype)
        
        output = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.accelerator.device),
            input_ids=input_ids.to(self.accelerator.device),
            text_modulation_embeds=text_modulation_embeds,
            attention_mask=attention_mask.to(self.accelerator.device),
        )
        model_pred = output[0]

        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas.to(dtype=model_pred.dtype)) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise.float() - model_input.float()

        loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1)
        loss = loss.mean()

        return loss

    def adafusedit_training_step(self, batch):
        """AdaFuseDiT training step."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(self.accelerator.device)
        bsz = input_ids.shape[0]

        # Use precomputed position_ids and attention_mask
        #position_ids = self._cached_position_ids.expand(bsz, -1)
        #llm_attention_mask = self._cached_llm_attn_mask.expand(bsz, -1, -1, -1)
        position_ids = torch.arange(input_ids.shape[1], device=self.accelerator.device).unsqueeze(0)
        with torch.no_grad():
            llm_output = self.llm(
                input_ids.to(self.accelerator.device),
                attention_mask,
                position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=False,
            )
        all_hidden_states = llm_output[1]
        
        text_hidden_states_num = getattr(self.hparams.model, 'text_hidden_states_num', 1)
        
        if text_hidden_states_num > 1:
            text_hidden_states = [
                all_hidden_states[-text_hidden_states_num + i].to(dtype=self.weight_dtype)
                for i in range(text_hidden_states_num)
            ]
        else:
            text_hidden_states_index = getattr(self.hparams.model, 'text_hidden_states_index', -1)
            text_hidden_states = all_hidden_states[text_hidden_states_index].to(dtype=self.weight_dtype)

        model_input = pixel_values.to(self.accelerator.device)
        with torch.no_grad():
            model_input = self.vae.encode(model_input.float()).latent_dist.sample()
        model_input = (model_input - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        #model_input = model_input.to(dtype=self.weight_dtype)

        noise = torch.randn_like(model_input)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.hparams.trainer.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.hparams.trainer.logit_mean,
            logit_std=self.hparams.trainer.logit_std,
            mode_scale=self.hparams.trainer.mode_scale,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=torch.float32).to(self.accelerator.device)
        noisy_model_input = (1.0 - sigmas) * model_input.float() + sigmas * noise.float()
        noisy_model_input = noisy_model_input.to(dtype=self.weight_dtype)

        model_pred = self.model(
            hidden_states=noisy_model_input,
            timestep=timesteps.to(self.accelerator.device),
            text_hidden_states=text_hidden_states,
            attention_mask=attention_mask,
        )

        if self.hparams.trainer.precondition_outputs:
            model_pred = model_pred * (-sigmas.to(dtype=model_pred.dtype)) + noisy_model_input

        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.hparams.trainer.weighting_scheme, sigmas=sigmas)

        if self.hparams.trainer.precondition_outputs:
            target = model_input
        else:
            target = noise.float() - model_input.float()

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
        )
        loss = loss.mean()

        return loss

    def log(self, data, progress_bar=True):
        """Log metrics."""
        if self.accelerator.is_main_process:
            if is_wandb_available():
                wandb.log(data, step=self.global_step)
            if progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix(data)

    @torch.no_grad()
    def update_ema(self):
        """Update the EMA model."""
        if self.ema is None:
            return
        if self.hparams.ema.update_steps is not None and self.global_step % self.hparams.ema.update_steps == 0:
            self.accelerator.wait_for_everyone()
            
            ema_params = OrderedDict(self.ema.named_parameters())
            model_params = OrderedDict(self.accelerator.unwrap_model(self.model).named_parameters())
            assert set(ema_params.keys()) == set(model_params.keys())

            for name, param in [(k, v) for k, v in model_params.items() if v.requires_grad]:
                ema_params[name].mul_(self.hparams.ema.decay).add_(param.data, alpha=1 - self.hparams.ema.decay)

    @abstractmethod
    def save_checkpoint(self):
        pass

    def train(self):
        """Main training loop."""
        self.before_training()
        
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.hparams.trainer.gradient_accumulation_steps)
        num_train_epochs = math.ceil(self.hparams.trainer.max_steps / num_update_steps_per_epoch)
        
        starting_epoch = self.global_step // num_update_steps_per_epoch
        resume_step = self.global_step % num_update_steps_per_epoch

        profile_stats = {}

        for epoch in range(starting_epoch, num_train_epochs):
            if epoch == starting_epoch and resume_step > 0:
                active_dataloader = self.accelerator.skip_first_batches(
                    self.train_dataloader, 
                    resume_step * self.hparams.trainer.gradient_accumulation_steps
                )
                if self.accelerator.is_main_process:
                    print(f"⏭️ Skipping the first {resume_step * self.hparams.trainer.gradient_accumulation_steps} batches")
            else:
                active_dataloader = self.train_dataloader
            
            t0 = time.time()
            data_iter = iter(active_dataloader)
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"⚠️ Skipping a batch (data issue): {e}")
                    t0 = time.time()
                    continue

                profile_stats["Data"] = time.time() - t0
                
                do_profile = self.accelerator.is_main_process and (self.global_step % 10 == 0)

                with self.accelerator.accumulate(self.model):
                    with ProfileTimer("Forward", profile_stats, do_profile):
                        loss = self.training_step(batch)
                    with ProfileTimer("Backward", profile_stats, do_profile):
                        self.backward(loss)
                    with ProfileTimer("Optim", profile_stats, do_profile):
                        grad_norm=self.optimizer_step()

                if do_profile:
                    sync_status = "Sync" if self.accelerator.sync_gradients else "NoSync"
                    msg = f"⏱️ [{sync_status}][Step {self.global_step}] " + " | ".join([f"{k}: {v:.3f}s" for k, v in profile_stats.items()])
                    print(msg)

                if self.accelerator.sync_gradients:
                    self.global_step += 1

                    if self.global_step % self.hparams.trainer.logging_steps == 0:
                        log_data = {
                            "train/loss": loss.detach().item(), 
                            "train/lr": self.lr_scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/grad_norm": grad_norm,
                        }
                        if do_profile and len(profile_stats) > 0:
                            for k, v in profile_stats.items():
                                log_data[f"profile/{k.lower()}"] = v
                        self.log(log_data)
                    elif self.accelerator.is_main_process:
                        self.progress_bar.update(1)

                    self.update_ema()
                    self.save_checkpoint()

                    if self.global_step >= self.hparams.trainer.max_steps:
                        break
                
                t0 = time.time()
            
            if self.global_step >= self.hparams.trainer.max_steps:
                break

        self.after_training()

    def after_training(self):
        """Cleanup after training."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            print("Training completed!")
        self.accelerator.end_training()


class AccelerateTrainer(Trainer):
    """Trainer built on HuggingFace Accelerate."""
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.global_step = 0

        project_config = ProjectConfiguration(
            project_dir=hparams.trainer.checkpoint_dir,
            logging_dir=os.path.join(hparams.trainer.checkpoint_dir, "logs")
        )
        
        mixed_precision = hparams.trainer.mixed_precision
        if mixed_precision == "fp32":
            mixed_precision = "no"


       
        self.accelerator = Accelerator(
            gradient_accumulation_steps=hparams.trainer.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb" if is_wandb_available() else None,
            project_config=project_config,
        )

        if hparams.trainer.seed is not None:
            set_seed(hparams.trainer.seed)

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.total_batch_size = (
            hparams.data.batch_size * 
            self.accelerator.num_processes * 
            hparams.trainer.gradient_accumulation_steps
        )

        if self.accelerator.is_main_process:
            print(f"🚀 Initializing AccelerateTrainer...")
            print(f"  - processes: {self.accelerator.num_processes}")
            print(f"  - mixed precision: {self.accelerator.mixed_precision}")

        if self.accelerator.is_main_process:
            print(f"📦 Building model...")
        self.model = build_model(hparams)
        self.model.train()
        # Count parameters for logging/wandb
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._param_stats = {
            "params/total": total_params,
            "params/trainable": trainable_params,
            "params/total_m": total_params / 1e6,
            "params/trainable_m": trainable_params / 1e6,
        }
        print(f"  - Parameters: total={total_params:,} ({total_params / 1e6:.2f}M), trainable={trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        if hparams.trainer.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(), 
            **hparams.optimizer
        )

        self.lr_scheduler = get_scheduler(
            **hparams.lr_scheduler, 
            optimizer=self.optimizer, 
            num_training_steps=hparams.trainer.max_steps * hparams.trainer.gradient_accumulation_steps
        )

        if self.accelerator.is_main_process:
            print(f"📚 Loading dataset...")
        self.train_dataloader = get_dataloader(hparams)

        if self.accelerator.is_main_process:
            print(f"⚙️ Preparing distributed training...")
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        device = self.accelerator.device
        if self.accelerator.is_main_process:
            print(f"📦 Loading VAE to {device}...")
        self.vae = AutoencoderKL.from_pretrained(**hparams.vae)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(device, dtype=torch.float32)

        if hparams.model.name == "DiT":
            if hparams.model.encoder_type == "llm":
                if self.accelerator.is_main_process:
                    print(f"📦 Loading LLM to {device}...")
                self.llm = get_llm(hparams.model.base, self.accelerator.unwrap_model(self.model).config.base_config)
                self.llm.requires_grad_(False)
                self.llm.eval()
                self.llm.to(device, dtype=self.weight_dtype)
            else:
                raise ValueError(f"Unknown encoder type: {hparams.model.encoder_type}")
            self.training_step = self.dit_training_step
            
        elif hparams.model.name == "AdaFuseDiT":
            if hparams.model.encoder_type == "llm":
                if self.accelerator.is_main_process:
                    print(f"📦 Loading LLM to {device}...")
                self.llm = get_llm(hparams.model.base, self.accelerator.unwrap_model(self.model).config.base_config)
                self.llm.requires_grad_(False)
                self.llm.eval()
                self.llm.to(device, dtype=self.weight_dtype)
                
                # Precompute position_ids and attention_mask for the LLM
                max_seq_len = hparams.data.max_prompt_length 
                
                self._cached_position_ids = torch.arange(max_seq_len, device=device).unsqueeze(0)
                self._cached_llm_attn_mask = update_self_attention_mask(
                    torch.ones(1, max_seq_len, device=device, dtype=torch.long),
                    0, False, device, dtype=self.weight_dtype
                )
                if self.accelerator.is_main_process:
                    print(f"  ✅ Precomputed position_ids and attention_mask, seq_len: {max_seq_len}")
            else:
                raise ValueError(f"Unknown encoder type: {hparams.model.encoder_type}")
            self.training_step = self.adafusedit_training_step
            
        elif hparams.model.name == "FuseDiT":
            if hparams.model.encoder_type == "clip-llm":
                if self.accelerator.is_main_process:
                    print(f"📦 Loading CLIP to {device}...")
                self.clip = CLIPTextModelWithProjection.from_pretrained(**hparams.clip_l)
                self.clip.requires_grad_(False)
                self.clip.eval()
                self.clip.to(device, dtype=self.weight_dtype)
            self.training_step = self.fusedit_training_step
        else:
            raise ValueError(f"Unknown model name: {hparams.model.name}")

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(**hparams.noise_scheduler)

        if hparams.ema.update_steps is not None:
            if self.accelerator.is_main_process:
                print(f"📦 Creating EMA model...")
            self.ema = deepcopy(self.accelerator.unwrap_model(self.model))
            self.ema.requires_grad_(False)
            self.ema.to(device)
        else:
            self.ema = None

        self.accelerator.wait_for_everyone()
        self.load_checkpoint()

        if self.accelerator.is_main_process:
            print(f"✅ AccelerateTrainer initialized")
            print(f"  - device: {self.accelerator.device}")
            print(f"  - processes: {self.accelerator.num_processes}")
            print(f"  - mixed precision: {self.accelerator.mixed_precision}")
            print(f"  - grad accumulation steps: {hparams.trainer.gradient_accumulation_steps}")
            print(f"  - total batch size: {self.total_batch_size}")

    def load_checkpoint(self):
        """Load checkpoint."""
        if self.hparams.trainer.resume_from is None:
            return
            
        resume_from = self.hparams.trainer.resume_from
        
        if resume_from == "latest":
            checkpoint_dir = self.hparams.trainer.checkpoint_dir
            if os.path.exists(checkpoint_dir):
                checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    resume_from = os.path.join(checkpoint_dir, checkpoints[-1])
                else:
                    if self.accelerator.is_main_process:
                        print("⚠️ No checkpoint found, starting from scratch")
                    return
            else:
                if self.accelerator.is_main_process:
                    print("⚠️ Checkpoint directory missing, starting from scratch")
                return
        
        if os.path.exists(resume_from):
            if self.accelerator.is_main_process:
                print(f"📂 Resuming from {resume_from}...")
            
            self.accelerator.load_state(resume_from)
            self.global_step = int(os.path.basename(resume_from).split("-")[1])
            
            ema_path = os.path.join(resume_from, "ema.pt")
            if self.ema is not None and os.path.exists(ema_path):
                self.ema.load_state_dict(torch.load(ema_path, map_location=self.accelerator.device))
                if self.accelerator.is_main_process:
                    print(f"✅ EMA model loaded from {ema_path}")
            
            if self.accelerator.is_main_process:
                print(f"✅ Resumed from step {self.global_step}")
        else:
            if self.accelerator.is_main_process:
                print(f"⚠️ Checkpoint path {resume_from} does not exist, starting from scratch")

    def save_checkpoint(self):
        """Save checkpoint."""
        if self.global_step % self.hparams.trainer.checkpointing_steps != 0:
            return
        
        self.accelerator.wait_for_everyone()
            
        save_path = os.path.join(
            self.hparams.trainer.checkpoint_dir, 
            f"checkpoint-{self.global_step}"
        )
        
        self.accelerator.save_state(save_path)
        self.accelerator.wait_for_everyone()
        
        if self.ema is not None and self.accelerator.is_main_process:
            ema_path = os.path.join(save_path, "ema.pt")
            torch.save(self.ema.state_dict(), ema_path)
        
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).config.save_pretrained(save_path)
            print(f"💾 Checkpoint saved to {save_path}")
        
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            checkpoints = [d for d in os.listdir(self.hparams.trainer.checkpoint_dir) 
                          if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            checkpoints_to_keep = set()
            for ckpt in checkpoints:
                step = int(ckpt.split("-")[1])
                if step % self.hparams.trainer.consolidation_steps == 0:
                    checkpoints_to_keep.add(ckpt)
            
            if checkpoints:
                checkpoints_to_keep.add(checkpoints[-1])
            
            for ckpt in checkpoints:
                if ckpt not in checkpoints_to_keep:
                    ckpt_path = os.path.join(self.hparams.trainer.checkpoint_dir, ckpt)
                    shutil.rmtree(ckpt_path)

    def backward(self, loss):
        """Backpropagation."""
        self.accelerator.backward(loss)

    def optimizer_step(self):
        """Optimizer update."""
        if self.accelerator.sync_gradients:
            grad_norm=self.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.hparams.trainer.gradient_clipping
            )
        
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm.item() if grad_norm is not None else None


def get_trainer(hparams):
    """Get a trainer instance."""
    return AccelerateTrainer(hparams)
