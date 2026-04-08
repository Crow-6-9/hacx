# Parameter Golf: Training Steps to Get a Model

Follow these steps to train your model for the OpenAI Parameter Golf challenge. Depending on your hardware, you can either train locally (if you have an Apple Silicon Mac or sufficient local GPU) or run it on a remote GPU instance via Runpod.

## Prerequisites
- A Python environment (Python 3.8+ recommended).
- Git installed.

---

## 1. Local Setup and Environment Preparation

First, set up your project environment:

```bash
# Clone the repository (if you haven't already in your workspace)
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

---

## 2. Install Dependencies

Depending on whether you are running on an Apple Silicon device or a standard CUDA/CPU setup, install the appropriate packages:

**For standard setups (or Cloud GPUs):**
```bash
pip install -r requirements.txt
```

**For Apple Silicon (Mac):**
```bash
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

---

## 3. Download the Dataset

You need to download the FineWeb validation set and training shards using the `sp1024` variant.

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

*Note: By default, this downloads 10 training shards. For a smaller "smoke test" subset to quickly see if it works, you can pass `--train-shards 1`.*

This will populate:
- `./data/datasets/fineweb10B_sp1024/`
- `./data/tokenizers/`

---

## 4. Start Model Training

Depending on your environment, launch the training script.

### Option A: Local Training on Mac (Apple Silicon)
Use the MLX-based training script:

```bash
# These are environment variables structured for Mac/Linux shells. 
# On Windows PowerShell, set the environment variables via `$env:RUN_ID="mlx_smoke"` etc. before running.
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python train_gpt_mlx.py
```

### Option B: Training on a Local or Remote GPU (e.g., 1xH100/NVIDIA GPU) 
If you are running on a machine with a CUDA-enabled GPU:

```bash
# For a single GPU node:
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

---

## 5. View Your Results

At the completion of the `train_gpt.py` or `train_gpt_mlx.py` run:
1. The script will output a final report including `val_loss`, `val_bpb`, and the compressed model size (in bytes).
2. The final trained components will typically be summarized as part of the `final_int8_zlib_roundtrip` metrics to ensure your model falls within the < 16MB required limit for the challenge.

**Note from the rules:** Your evaluation step, to be a valid entry, shouldn't access training data during evaluation, and your entire script size + weights combined cannot exceed 16,000,000 bytes.
