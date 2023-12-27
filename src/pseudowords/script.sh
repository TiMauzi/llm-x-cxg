#!/bin/bash
#
#SBATCH --job-name=pseudowords
#SBATCH --comment="Training of pseudoword embeddings."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.sockel@campus.lmu.de
#SBATCH --chdir=/home/s/sockel/Desktop/llm-x-cxg/src/pseudowords
#SBATCH --output=/home/s/sockel/Desktop/llm-x-cxg/out/slurm.%A.%a.%j.%N.out
#SBATCH --array=1-15

total_tasks=${SLURM_ARRAY_TASK_COUNT}

task_range=$((562 / total_tasks))

start=$((($SLURM_ARRAY_TASK_ID - 1) * task_range))
end=$((($SLURM_ARRAY_TASK_ID * task_range)))

python3 -u get_bert_kee_pseudowords_avg.py --device="cuda" --start=$start --end=$end 2>&1 | tee -a ../../out/cache/log$SLURM_ARRAY_TASK_ID.txt