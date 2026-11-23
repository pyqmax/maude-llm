#!/bin/bash
# One-time setup on Fir (login node): project folders, venv, embedding model.
set -euo pipefail
ROOT=/path/to/maude-coding
mkdir -p "$ROOT"/{scripts,data,logs,results,models}
cd "$ROOT"
module load StdEnv/2023 python/3.11.5 arrow/24.0.0
if [ ! -d .venv ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install -q --no-index --upgrade pip
pip install -q --no-index torch sentence-transformers rank_bm25 pandas pyarrow requests huggingface_hub
python -c "import torch, sentence_transformers, rank_bm25, pandas; print('torch', torch.__version__, 'st', sentence_transformers.__version__)"
# Compute nodes have no internet: fetch the embedding model here.
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-base-en-v1.5', local_dir='$ROOT/models/bge-base-en-v1.5'))"
echo SETUP_DONE
