#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=maude-shortlist
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/shortlist-%j.out
#SBATCH --error=logs/shortlist-%j.err

set -euo pipefail
cd /path/to/maude-coding
module load StdEnv/2023 python/3.11.5 arrow/24.0.0
source .venv/bin/activate
python -P scripts/build_shortlists.py data/clean/reports.parquet data/sample \
    data/codebook_A_E.tsv models/bge-base-en-v1.5 results/shortlists
echo JOB_DONE
