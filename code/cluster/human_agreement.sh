#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=maude-human
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/human-%j.out
#SBATCH --error=logs/human-%j.err

set -euo pipefail
cd /path/to/maude-coding
module load StdEnv/2023 python/3.11.5 arrow/24.0.0
source .venv/bin/activate
python -P scripts/human_agreement.py data/clean/reports.parquet results/human_agreement
echo JOB_DONE
