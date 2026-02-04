"""
Evaluate DrawBench generations with the unified reward prompt in ``unifiedreward2.py``.

Usage:
  python evaluation/eval_drawbench_unifiedreward2.py \
      --images-dir /path/to/draw-7-28 \
      --metadata-file /path/to/metadata.json \
      --server-url http://localhost:8080 \
      --output-file results.jsonl \
      --batch-size 8
"""
import argparse
import json
import os
from typing import List

from tqdm import tqdm

# Reuse the problem template and batching interface used in unifiedreward2.py
from evaluation.unifiedreward2 import evaluate_batch


PROBLEM_TEMPLATE = (
    "You are presented with a generated image and its associated text caption. "
    "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
    "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    "Text Caption: [{caption}]"
)


def _load_captions(metadata_path: str) -> List[str]:
    """Load captions from the DrawBench metadata file."""
    with open(metadata_path, "r") as f:
        data = json.load(f)

    # sample_drawbench.py collects metadata.values() where each item is expected to
    # contain a 'caption' field.
    captions = []
    if isinstance(data, list):
        iterator = data
    elif isinstance(data, dict):
        iterator = data.values()
    else:
        raise ValueError(f"Unsupported metadata format: {type(data)}")

    for item in iterator:
        if isinstance(item, dict):
            caption = (
                item.get("caption")
                or item.get("prompt")
                or item.get("text")
                or item.get("instruction")
            )
        else:
            caption = str(item)

        if caption is None:
            raise ValueError("Missing caption in metadata entry.")
        captions.append(caption)

    return captions


def _collect_samples(images_dir: str, captions: List[str]):
    """Collect generated images and pair them with captions based on filename index."""
    samples = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        # sample_drawbench.py uses the pattern {index}_sample_{i}.png
        try:
            idx_str = fname.split("_")[0]
            idx = int(idx_str)
        except Exception:
            continue

        caption = captions[idx] if idx < len(captions) else ""
        samples.append(
            {
                "image_path": os.path.join(images_dir, fname),
                "caption": caption,
            }
        )
    return samples


def evaluate_drawbench(images_dir: str, metadata_file: str, server_url: str, output_file: str, batch_size: int):
    captions = _load_captions(metadata_file)
    samples = _collect_samples(images_dir, captions)
    if not samples:
        raise ValueError(f"No images found in {images_dir}")

    results = []
    for start in tqdm(range(0, len(samples), batch_size), desc="Evaluating"):
        batch = samples[start : start + batch_size]
        input_data = []
        for s in batch:
            problem = PROBLEM_TEMPLATE.format(caption=s["caption"])
            input_data.append({"problem": problem, "images": [s["image_path"]]})

        outputs = evaluate_batch(input_data, server_url, image_root=None)
        for s, out in zip(batch, outputs):
            results.append(
                {
                    "image_path": s["image_path"],
                    "caption": s["caption"],
                    "model_output": out["model_output"],
                }
            )

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} evaluation records to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DrawBench generations with unifiedreward2 prompt.")
    parser.add_argument("--images-dir", required=True, help="Directory containing generated images (e.g., draw-7-28).")
    parser.add_argument("--metadata-file", required=True, help="DrawBench metadata JSON used for generation.")
    parser.add_argument("--server-url", default="http://localhost:8080", help="UnifiedReward2 server endpoint.")
    parser.add_argument("--output-file", default="drawbench_unifiedreward2.jsonl", help="Where to save evaluation results.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of images per request batch.")
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_drawbench(
        images_dir=args.images_dir,
        metadata_file=args.metadata_file,
        server_url=args.server_url,
        output_file=args.output_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
