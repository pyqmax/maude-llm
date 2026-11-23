#!/bin/bash
#SBATCH --account=ACCOUNT
#SBATCH --job-name=maude-medgemma
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/medgemma-%j.out
#SBATCH --error=logs/medgemma-%j.err
#
# Self-hosted MedGemma 4B IT on the stage 1 plan (unseen set, arms 0/A/B/C;
# comparison set, arms 0/A). Resumable: rerun the same command after a time-out.

set -euo pipefail
cd /path/to/maude-coding
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a arrow/24.0.0 cuda/12.6
source /path/to/tomi-new/.venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -P scripts/run_medgemma.py /path/to/models/medgemma-4b-it \
    data/medgemma_jobs.jsonl results/medgemma_results.jsonl 48000 600
echo JOB_DONE
