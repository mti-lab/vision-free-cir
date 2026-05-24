# Towards Vision-Free CIR: Attribute-Augmented Scoring and LLM-Based Reranking for Zero-Shot Composed Image Retrieval

[![SIGIR 2026 Workshop SynthIR](https://img.shields.io/badge/SIGIR%202026%20Workshop-SynthIR-brightgreen)](https://zdzheng.xyz/SIGIR2026Workshop-SynthIR)
[![Best Paper](https://img.shields.io/badge/Award-Best%20Paper-gold)](https://zdzheng.xyz/SIGIR2026Workshop-SynthIR)
[![arXiv](https://img.shields.io/badge/arXiv-2607.12621-b31b1b)](https://arxiv.org/abs/2607.12621)

Official implementation of the paper **"Towards Vision-Free CIR: Attribute-Augmented Scoring and LLM-Based Reranking for Zero-Shot Composed Image Retrieval"**.

- 📄 Paper: [arXiv:2607.12621](https://arxiv.org/abs/2607.12621)
- 🏆 Venue: [SIGIR 2026 Workshop on Synthetic Data for Information Retrieval (SynthIR)](https://zdzheng.xyz/SIGIR2026Workshop-SynthIR) — **Best Paper Award**

## Abstract

We propose a Vision-Free framework for Composed Image Retrieval (CIR) that operates entirely in the text space by converting images into textual representations. To address the information loss inherent in this conversion, the framework introduces two techniques: **(1) Attribute-Augmented Hybrid Scoring**, which compensates for lost visual details via explicit attribute matching, and **(2) LLM-Based Reranking**, which verifies the semantic consistency of top candidates. On CIRR, the framework yields a gain of +8.79%, while results on FashionIQ reveal a trade-off between semantic reasoning and fine-grained visual matching.

## Setup
1. Clone [fashionIQ repository](https://github.com/XiaoxiaoGuo/fashion-iq) to `./fashion-iq`
2. Clone [fashionIQ metadata repository](https://github.com/hongwang600/fashion-iq-metadata) to `./fashion-iq-metadata`
3. Install dependencies: `uv sync`
4. Create `.env` file and add your OpenAI API key
5. Download images to local storage:
   ```bash
   uv run python src/download_images.py dress
   uv run python src/download_images.py shirt
   uv run python src/download_images.py toptee
   ```
   Images will be saved to `images/{category}/` directory.

## Usage
Run: `uv run python src/main.py config/your_config.yaml`. Config files are located in `config/` directory.

Config files follow the pattern `{dataset}_{split}_{ablation}.yaml`:

- **dataset**: `cirr` or `fashioniq_{dress,shirt,toptee}`
- **split**: `val` (full evaluation) or `train` (grid search for hyperparameters)
- **ablation**:
  - `dense` — Dense retrieval only
  - `dense_attr` — Dense + attribute matching
  - `dense_rerank` — Dense + LLM reranking
  - `dense_attr_rerank` — Full pipeline (dense + attribute + reranking)

Example: `config/cirr_val_dense_attr_rerank.yaml` runs the full pipeline on the CIRR validation set.

## Attribute Discovery
Discover dataset-specific attributes from the train split using an LLM. The script samples images, proposes attributes per image, selects the top-N, extracts values, and consolidates them into a categorical schema.

```bash
# FashionIQ (per category)
uv run python src/attribute_discovery.py --dataset fashioniq --category dress

# CIRR
uv run python src/attribute_discovery.py --dataset cirr
```

Optional flags: `--num_samples` (images for proposal, default 50), `--num_attributes` (final attribute count, default 4), `--num_value_samples` (images for value extraction, default 100).

Outputs are written to `discovered_attributes/{dataset_id}_attribute_schema.json` (plus intermediate proposal/extraction files). Copy the resulting schema into `src/schema/{dataset_id}.py` so that `main.py` and `grid_search.py` can consume it.

## Grid Search
Search for the optimal `alpha` (dense vs. attribute weight) and per-attribute weights on the train split. Captions, attributes, modified captions, and embeddings are cached under `precomputed/`.

```bash
uv run python src/grid_search.py config/fashioniq_dress_train.yaml
```

Train configs (`config/{dataset}_{category}_train.yaml`) define:
- `num_sample_queries`, `num_sample_database_images` — sampling sizes
- `grid_search.alpha_values` — list of alphas to sweep
- `grid_search.attribute_weight_configs` — named per-attribute weight combinations

Results are saved to `results/{exp_name}_results.json`, sorted by R@50. Use the best configuration to fill in the corresponding `*_val_*.yaml` config for full evaluation.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{shimada2026visionfreecir,
  title     = {Towards Vision-Free CIR: Attribute-Augmented Scoring and LLM-Based Reranking for Zero-Shot Composed Image Retrieval},
  author    = {Ryotaro Shimada and Yu-Chieh Lin and Yuji Nozawa and Youyang Ng and Osamu Torii and Yusuke Matsui},
  booktitle = {Proceedings of the SIGIR 2026 Workshop on Synthetic Data for Information Retrieval (SynthIR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.12621}
}
```
