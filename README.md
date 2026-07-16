# Towards Robust Image Watermarking via Diffusion Models Against Multi-turn Editing
This repository hosts the official PyTorch implementation of the paper: "Towards Robust Image Watermarking via Diffusion Models Against Multi-turn Editing" (Accepted by TMM 2026).

# About
We propose Reinforcement-Optimized Diversified Watermarking (RODW), a novel framework to tackle the unexplored challenge of maintaining watermark robustness under multi-turn image editing. RODW guides the watermarking model to better learn the complex characteristics of multi-turn image editing by simulating real-world instruction-driven image editing scenarios. Specifically, RODW introduces the diversified sampling generation strategy that constructs editing instructions in various styles. These instructions provide richer editing features for watermarking model training, enabling the watermarking model to learn from a more comprehensive set of simulated editing operations. Furthermore, we design a Reinforcement Learning Optimization (RLO) strategy to guide the watermark embedding process, which allows the watermarking model to learn fine-grained editing features, ultimately enhancing watermark robustness under multi-turn image editing.

# Getting Started
# Prerequisites
conda create -n rodw python=3.8
conda activate rodw
pip install -r requirements.txt

# construct vlm prompt 
For generate vlm prompt, you can run,

python ./get_refactoring_prompt.py

# Train

1. Download the [datasets](https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered) and put them into the data dir ./data.

2. Configure the train script and then run it.

```
bash train.sh
```
# Inference

1. Put your original image in ./path.
2. Put your prompt dataset in ./name.txt
3. For generalization testing, you can down [InstructDiffusion](https://github.com/cientgu/InstructDiffusion) and [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush).
4. Configure the test script and then run it.

```
bash inference.sh
```

# Acknowledgements
We borrow the code from [Robust-Wide Watermark](https://github.com/hurunyi/Robust-Wide). We appreciate the authors for sharing their code.

# Citation
If you find our work useful for your research, please consider citing our paper:

