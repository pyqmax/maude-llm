#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=maude-retrieval
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/retrieval-%j.out
#SBATCH --error=logs/retrieval-%j.err

set -euo pipefail
ROOT=/path/to/maude-coding
cd "$ROOT"
module load StdEnv/2023 python/3.11.5 arrow/24.0.0
source .venv/bin/activate
# -P, not -I: -I would ignore PYTHONPATH, which is how the arrow module provides pyarrow.
python -P scripts/retrieval_eval.py data/sample data/codebook_A_E.tsv \
    results/retrieval_eval_comparison.tsv comparison models/bge-base-en-v1.5
echo JOB_DONE
