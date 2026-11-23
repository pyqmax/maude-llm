#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=maude-prior
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prior-%j.out
#SBATCH --error=logs/prior-%j.err

set -euo pipefail
cd /path/to/maude-coding
module load StdEnv/2023 python/3.11.5 arrow/24.0.0
source .venv/bin/activate
python -P scripts/prior_eval.py data/clean/reports.parquet data/sample results/prior_eval.tsv
echo JOB_DONE
