#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=maude-medgemma-mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/medgemma-mig-%j.out
#SBATCH --error=logs/medgemma-mig-%j.err
#
# Same run as run_medgemma.sh on a 40 GB MIG slice, submitted in parallel because
# full-H100 queues are long. Writes to its own results file; whichever job starts
# first is kept and the other cancelled; files are merged by id afterwards.

set -euo pipefail
cd /path/to/maude-coding
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a arrow/24.0.0 cuda/12.6
source /path/to/tomi-new/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -P scripts/run_medgemma.py /path/to/models/medgemma-4b-it \
    data/medgemma_jobs.jsonl results/medgemma_results_mig.jsonl 32000 600
echo JOB_DONE
