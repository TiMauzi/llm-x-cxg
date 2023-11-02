#!/bin/bash
#
#SBATCH --job-name=pseudowords
#SBATCH --comment="Training of pseudoword embeddings."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.sockel@campus.lmu.de
#SBATCH --chdir=/home/s/sockel/Desktop/llm-x-cxg/src/pseudowords
#SBATCH --output=/home/s/sockel/Desktop/llm-x-cxg/out/slurm.%j.%N.out
#SBATCH --nodes=15  # Set the number of nodes
#SBATCH --ntasks-per-node=1  # Set the number of tasks per node
#SBATCH --gres=gpu:1  # Request 1 GPU per task

# Load necessary modules and activate your virtual environment if needed
module load python
conda activate llm-cxg

# Define the total number of tasks and loop over them
total_tasks=843  # Change this to the number of tasks you want

for task_id in $(seq 0 $((total_tasks - 1))); do
    # Launch the Python script with the task-specific input
    python3 -u par_get_kee_pseudowords_avg.py --task_id $task_id &
done

# Wait for all tasks to complete
wait
