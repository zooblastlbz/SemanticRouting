import random
from pathlib import Path
import os
from typing import Dict, Any, List

from datasets import load_dataset
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.distributed as dist
from torchvision import transforms
from transformers import AutoTokenizer


def _get_process_rank():
    """Get the current process rank."""
    try:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _is_main_process():
    """Return True if this is the main process."""
    return _get_process_rank() == 0


class PreprocessTrain:
    """
    On-the-fly preprocessing used with dataset.with_transform().
    Mirrors the PreprocessTrain logic in tmp.py.
    """
    def __init__(
        self,
        image_key: str,
        text_key: str,
        tokenizer,
        max_length: int,
        train_transforms,
        instruction: str = '',
        apply_chat_template: bool = False,
        add_generation_prompt: bool = False,
        random_dropping_rate: float = 0.0,
        image_root: str = None,
    ):
        self.image_key = image_key
        self.text_key = text_key
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_transforms = train_transforms
        self.instruction = instruction
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.random_dropping_rate = random_dropping_rate
        self.image_root = Path(image_root) if image_root else None
        
        # Track failures
        self.failed_count = 0
        self.total_count = 0

    def _create_fallback_image(self, target_size=(256, 256)):
        """Create a default RGB placeholder image."""
        return Image.new('RGB', target_size, color=(128, 128, 128))

    def _load_image_safely(self, image_item):
        """Load an image safely and fall back if needed."""
        try:
            if isinstance(image_item, str):
                if self.image_root:
                    image_item = str(self.image_root / image_item)
                
                if not os.path.exists(image_item):
                    return self._create_fallback_image()
                
                with Image.open(image_item) as img:
                    image = img.convert("RGB")
                    image.load()
                    return image
            else:
                if hasattr(image_item, 'convert'):
                    image = image_item.convert("RGB")
                    image.load()
                    return image
                return self._create_fallback_image()
        except Exception:
            self.failed_count += 1
            return self._create_fallback_image()

    def _apply_prompt_drop(self, captions: List[str]) -> List[str]:
        """Randomly drop prompts with a given probability."""
        if self.random_dropping_rate <= 0:
            return captions
        
        return [
            "" if random.random() < self.random_dropping_rate else caption
            for caption in captions
        ]

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        images = []
        valid_indices = []
        
        # Process images
        for i, image_item in enumerate(examples[self.image_key]):
            self.total_count += 1
            image = self._load_image_safely(image_item)
            
            if image is not None:
                try:
                    transformed_image = self.train_transforms(image)
                    images.append(transformed_image)
                    valid_indices.append(i)
                except Exception:
                    try:
                        fallback_image = self._create_fallback_image()
                        transformed_image = self.train_transforms(fallback_image)
                        images.append(transformed_image)
                        valid_indices.append(i)
                        self.failed_count += 1
                    except Exception:
                        continue

        # If no valid images, create a default one
        if len(images) == 0:
            fallback_image = self._create_fallback_image()
            try:
                transformed_image = self.train_transforms(fallback_image)
                images.append(transformed_image)
                valid_indices.append(0)
            except Exception:
                images.append(torch.zeros(3, 256, 256))
                valid_indices.append(0)

        # Keep captions only for valid samples
        captions: List[str] = []
        original_captions = examples[self.text_key]
        
        for idx in valid_indices:
            if idx < len(original_captions):
                caption = original_captions[idx]
            else:
                caption = original_captions[0] if original_captions else ""
            
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if len(caption) > 0 else "")
            else:
                captions.append("")

        # Ensure images and captions lengths match
        min_length = min(len(images), len(captions))
        if min_length == 0:
            images = [torch.zeros(3, 256, 256)]
            captions = [""]
        
        images = images[:min_length]
        captions = captions[:min_length]

        # Apply prompt drop (for CFG)
        captions = self._apply_prompt_drop(captions)

        # Add instruction prefix to non-empty captions
        if self.instruction:
            captions = [self.instruction + caption  for caption in captions]

        # Tokenize captions
        if self.apply_chat_template:
            all_input_ids = []
            all_attention_masks = []
            
            for caption in captions:
                if caption:
                    tokenized = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": caption}],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        add_generation_prompt=self.add_generation_prompt,
                        return_dict=True,
                    )
                else:
                    tokenized = self.tokenizer(
                        "",
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                    )
                    tokenized = {
                        "input_ids": tokenized.input_ids,
                        "attention_mask": tokenized.attention_mask,
                    }
                
                all_input_ids.append(tokenized["input_ids"].squeeze(0))
                all_attention_masks.append(tokenized["attention_mask"].squeeze(0))
            
            input_ids = torch.stack(all_input_ids)
            attention_mask = torch.stack(all_attention_masks)
        else:
            inputs = self.tokenizer(
                captions,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

        examples["pixel_values"] = images
        examples["input_ids"] = input_ids
        examples["attention_mask"] = attention_mask
        return examples


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that filters invalid samples and stacks tensors.
    """
    # Drop samples that failed processing (None)
    examples = [e for e in examples if e is not None]
    
    if not examples:
        return None

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }


def get_dataloader(hparams):
    """
    Create a DataLoader for a local JSON dataset.

    Uses dataset.with_transform() for online processing.
    """
    tokenizer = AutoTokenizer.from_pretrained(hparams.data.tokenizer)
    
    image_key = getattr(hparams.data, 'image_key', 'image')
    text_key = getattr(hparams.data, 'text_key', 'text')
    
    instruction = getattr(hparams.data, 'instruction', '')
    apply_chat_template = getattr(hparams.data, 'apply_chat_template', False)
    add_generation_prompt = getattr(hparams.data, 'add_generation_prompt', False)
    random_dropping_rate = getattr(hparams.data, 'random_dropping_rate', 0.0)
    image_root = getattr(hparams.data, 'image_root', None)
    


    max_length = hparams.data.max_prompt_length 
    
    if _is_main_process():
        print(f"📚 Loading dataset {hparams.data.data_path}...")
    
    dataset = load_dataset('json', data_files=hparams.data.data_path, split='train')
    
    if _is_main_process():
        print(f"✅ Loaded {len(dataset)} samples")
        print(f"📌 Using keys: image_key='{image_key}', text_key='{text_key}'")
    
    if image_key not in dataset.column_names:
        raise KeyError(f"❌ Image key '{image_key}' not found in JSON")
    if text_key not in dataset.column_names:
        raise KeyError(f"❌ Text key '{text_key}' not found in JSON")
    
    # Build image transforms
    center_crop = getattr(hparams.data, 'center_crop', False)
    train_transforms = transforms.Compose([
        transforms.Resize(hparams.data.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(hparams.data.resolution) if center_crop else transforms.RandomCrop(hparams.data.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Create preprocessing wrapper
    preprocess_train = PreprocessTrain(
        image_key=image_key,
        text_key=text_key,
        tokenizer=tokenizer,
        max_length=max_length,
        train_transforms=train_transforms,
        instruction=instruction,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
        random_dropping_rate=random_dropping_rate,
        image_root=image_root,
    )
    
    # Apply with_transform for on-the-fly processing (matches tmp.py)
    train_dataset = dataset.with_transform(preprocess_train)
    
    if _is_main_process():
        print(f"✅ Using with_transform for on-the-fly preprocessing")
        print(f"  - max_length: {max_length}")
        print(f"  - apply_chat_template: {apply_chat_template}")
        print(f"  - random_dropping_rate: {random_dropping_rate}")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(int(hparams.trainer.seed) + _get_process_rank())

    num_workers = hparams.data.dataloader_num_workers
    persistent_workers = getattr(hparams.data, 'persistent_workers', True) and num_workers > 0
    prefetch_factor = getattr(hparams.data, 'prefetch_factor', 4) if num_workers > 0 else None
    pin_memory = getattr(hparams.data, 'pin_memory', True)
    timeout = getattr(hparams.data, 'dataloader_timeout', 30)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hparams.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        timeout=timeout,
    )
    
    if _is_main_process():
        print(f"✅ DataLoader ready with {len(dataloader)} batches")
        print(f"  - num_workers: {num_workers}")
        print(f"  - persistent_workers: {persistent_workers}")
        print(f"  - prefetch_factor: {prefetch_factor}")
        print(f"  - pin_memory: {pin_memory}")
        print(f"  - timeout: {timeout}")
    
    return dataloader
