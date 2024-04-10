#!/bin/bash
#
#SBATCH --job-name=pseudowords
#SBATCH --comment="Training of pseudoword embeddings."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --chdir=/path/to/dir/
#SBATCH --output=/path/to/dir/slurm.%A.%a.%j.%N.out
#SBATCH --array=1-15

total_tasks=${SLURM_ARRAY_TASK_COUNT}

task_range=$((562 / total_tasks))

start=$(((${SLURM_ARRAY_TASK_ID} - 1) * task_range))
end=$(((${SLURM_ARRAY_TASK_ID} * task_range)))

python3 -u get_bsb_bert_kee_pseudowords_avg.py --device="cuda:0" --start=$start --end=$end --temp=${SLURM_ARRAY_TASK_ID} 2>&1 | tee -a ../../out/cache/log${SLURM_ARRAY_TASK_ID}.txt