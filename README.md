# Feature Rivalry

Experiments on feature-level rivalry in Gemma-2-2B using Gemma Scope SAEs.

## Setup

**1. Clone**

```bash
git clone https://github.com/YOUR_USER/feature_rivalry.git
cd feature_rivalry
```

**Paperspace Gradient:** after each fresh machine / image reset:

```bash
cp .env.example .env
# edit .env — set HF_TOKEN=... (read token from Hugging Face)
chmod +x setup_paperspace.sh
./setup_paperspace.sh
```

Alternatively: `export HF_TOKEN=hf_...` then run `./setup_paperspace.sh`.

The script pins PyTorch to `2.5.1+cu121`, fixes NumPy, sets `LD_LIBRARY_PATH` for bundled CUDA libs, runs `huggingface-cli login`, and installs deps from `requirements.txt`. **Never commit `.env`** (it is gitignored).

**Other environments:** install manually:

```bash
pip install -r requirements.txt
```

**2. Gemma-2-2B access**

- Open [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) and accept the license.
- Log in: `huggingface-cli login` (token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
- The model downloads on first run (~5GB) to `~/.cache/huggingface/`.

**3. Verify**

```bash
python sanity_check.py
```

## Running

- **Debug (small split, fast check):** `python run.py --debug --exp 1`
- **Full pipeline:** `python run.py --exp all` (or `--exp 1`, `2`, `3` individually).
- **Progress only:** `python run.py --status`

Outputs go under `results/` (caches, pickles, figures). That directory is gitignored. After a debug run, remove `results/dataset_splits.json` and `results/popqa_entropy_cache.json` (and related checkpoints) before a full run so splits are rebuilt at scale.

## Configuration

See `config.py` for model name, layers, entropy thresholds, and prompt counts.
