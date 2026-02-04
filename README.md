
<!-- <img src="image/logo.png" width="38" height="38" alt="">  -->
<div align="center">

# SemanticRouting

### Exploring Multi-Layer LLM Feature Weighting for Diffusion Transformers

[![arXiv](https://img.shields.io/badge/arXiv-2602.03510-b31b1b.svg)](https://arxiv.org/abs/2602.03510)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/license/apache-2-0)
[![GitHub stars](https://img.shields.io/github/stars/zooblastlbz/SemanticRouting?style=social)](https://github.com/zooblastlbz/SemanticRouting)

**Bozhou Li, Yushuo Guan, Haolin Li, Bohan Zeng, Yiyan Ji, Yue Ding, Pengfei Wan, Kun Gai, Yuanxing Zhang, Wentao Zhang**

**Peking University | Kuaishou Technology | Fudan University | Nanjing University | UCAS**

**Preprint**

If this repository helps you, please give it a ⭐ for updates!

</div>

---

## 📋 Overview

**SemanticRouting** enhances text-to-image DiT models by introducing a dynamic **semantic routing** mechanism for multi-layer LLM features. Traditional methods rely on static or single-layer text conditioning, failing to account for the semantic hierarchy in LLMs and the non-stationary denoising dynamics of diffusion models (over time and network depth).

This repository implements a unified normalized convex fusion framework equipped with lightweight gates. We systematically explore:

*   **Time-wise Fusion**: Adapting fusion weights to the diffusion timestep $t$.
*   **Depth-wise Fusion**: Adapting weights to the DiT block index $d$.
*   **Joint Fusion**: Combining both time and depth adaptivity.

Our findings establish **Depth-wise Semantic Routing** as the superior strategy, significantly improving text-image alignment and compositional generation (e.g., **+9.97** on GenAI-Bench Counting), while purely time-wise fusion can suffer from trajectory mismatch issues during inference.

---

## ✨ Key Features

*   🚀 **Enhanced Alignment**: Delivers strong GenEval and GenAI-Bench results by leveraging hierarchical LLM semantics.
*   🧩 **Unified Framework**: Easily switch between fusion strategies (`uniform`, `static`, `time-wise`, `depth-wise`, `joint`) via simple config files.
*   🧠 **Semantic Routing**: Introduces learnable gating mechanisms to route information from appropriate LLM layers to specific DiT blocks.

---

## 📊 Performance

We evaluated our fusion strategies against strong baselines including Penultimate layer (B1), Uniform averaging (B2), Static fusion (B3), and FuseDiT.

### Overall Results

| Method | GenEval ↑ | GenAI-Bench ↑ | UnifiedReward ↑ |
| :--- | :---: | :---: | :---: |
| **Baselines** | | | |
| B1: Penultimate | 64.54 | 74.96 | 3.02 |
| B2: Uniform | 66.51 | 76.82 | 3.06 |
| B3: Static | 64.77 | 76.31 | 3.05 |
| **Deep-fusion Baseline** | | | |
| FuseDiT | 60.95 | 75.02 | 3.05 |
| **Our Strategies** | | | |
| S1: Time | 63.41 | 76.20 | 2.97 |
| **S2: Depth** | **67.07** | **79.07** | 3.06 |
| S3: Joint | 66.05 | 77.44 | 3.06 |

*Table 1: Comparison on GenEval, GenAI-Bench, and UnifiedReward. **S2 (Depth-wise)** achieves the best overall performance.*

---

## 🔬 Methodology

We introduce a unified formulation for multi-layer fusion. The final fused representation $H_{\text{cond}}(t,d)$ is formed via a softmax-normalized convex combination of normalized layer features:

$$ H_{\text{cond}}(t,d) = \sum_{l \in \mathcal{L}} \alpha_{t,d}^{(l)} \cdot \text{LN}(H^{(l)}) $$

Where the weights $\alpha_{t,d}^{(l)}$ are derived by applying a softmax function to learned logits $z_{t,d}$:

$$ \alpha_{t,d} = \mathrm{Softmax}(z_{t,d}) $$

We parameterize $z_{t,d}$ differently for each strategy:
*   **Time-wise**: $z_{t,d} = g_{\psi}(\phi(t))$ (Time-Conditioned Fusion Gate).
*   **Depth-wise**: $z_{t,d} = \beta_{d}$ (Block-specific learnable weights).
*   **Joint**: $z_{t,d} = g_{\psi_d}(\phi(t))$ (Depth-specific TCFG).

---

## 🚀 Installation

<details>
<summary><b>1. Clone & create environment (Python 3.12)</b></summary>

```bash
git clone https://github.com/zooblastlbz/SemanticRouting.git
cd SemanticRouting

conda create -n semanticrouting python=3.12 -y
conda activate semanticrouting
```

Install deps:

```bash
pip install -r requirements.txt
```

</details>

---

## 📁 Data Format

The repository expects data in **JSON** or **JSONL** format. Default keys are `image` and `text`, which can be overridden in config via `data.image_key` and `data.text_key`.

*   `image`: File path (resolved relative to `data.image_root` if provided).
*   `text`: String or list of strings.

**Example:**
```json
{
  "image": "images/sample_0001.jpg", 
  "text": "A scenic mountain lake at sunrise"
}
```

---

## 🏋️ Training

### 1. Select a Fusion Preset

Choose a configuration file from `configs/` to define your fusion strategy:

*   `configs/uniform.yaml`: Simple averaging of layers.
*   `configs/static.yaml`: Learnable global weights.
*   `configs/time-wise.yaml`: Time-dependent gating.
*   `configs/depth-wise.yaml`: Depth-dependent gating (**Recommended**).
*   `configs/joint.yaml`: Combined time and depth gating.

### 2. Launch Training

Run via the launcher script (set env vars as needed):
```bash
ACCELERATE_CONFIG=./accelerate_config.yaml \
CONFIG_FILE=./configs/depth-wise.yaml \
PYTHON_BIN=python \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29500 \
bash scripts/train.sh
```


---

## 🎨 Inference

### Export Pipeline

First, export the trained model and necessary components:

```bash
python utils/save_pipeline.py \
  --checkpoint /path/to/checkpoint-dir \
  --type adafusedit \
  --vae /path/to/vae \
  --scheduler /path/to/scheduler
```

### Run Generation

Generate images from text prompts:

```bash
python inference.py \
  --checkpoint /path/to/exported/pipeline \
  --prompt "A city skyline at dusk" \
  --resolution 512 \
  --num_inference_steps 25 \
  --guidance_scale 6.0
```

---

## 📈 Evaluation

Generate evaluation samples with Accelerate (for multi-GPU/multi-node) using the scripts in `evaluation/`:

- GenEval:
  ```bash
  accelerate launch evaluation/sample_geneval.py evaluation/geneval.yaml
  ```
- GenAIBench:
  ```bash
  accelerate launch evaluation/sample_genaibench.py evaluation/genaibench.yaml
  ```
- DrawBench:
  ```bash
  accelerate launch evaluation/sample_drawbench.py evaluation/drawbench.yaml
  ```

---

## 📝 Citation

If you use this code or the *SemanticRouting* paper, please cite:

```bibtex
@misc{li2026semanticroutingexploringmultilayer,
      title={Semantic Routing: Exploring Multi-Layer LLM Feature Weighting for Diffusion Transformers}, 
      author={Bozhou Li and Yushuo Guan and Haolin Li and Bohan Zeng and Yiyan Ji and Yue Ding and Pengfei Wan and Kun Gai and Yuanxing Zhang and Wentao Zhang},
      year={2026},
      eprint={2602.03510},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03510}, 
}
```

---

## 🙏 Acknowledgements

This codebase is adapted from and extends [tang-bd/fuse-dit](https://github.com/tang-bd/fuse-dit). We sincerely thank the original authors for their foundational work.

---

## 📄 License

This project is licensed under the Apache-2.0 License - see the [`LICENSE`](LICENSE) file for details.
