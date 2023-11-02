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

# Calculate total_tasks using a subshell and Python
total_tasks=$(python -c "import itertools
import json
with open('../../out/CoMaPP_all.json') as json_file:
    data = json.load(json_file)
data.sort(key=lambda x: x['label'])
data = [list(group) for _, group in itertools.groupby(data, key=lambda x: x['label'])]
print(len(data))")

echo "Total Tasks: $total_tasks" >> $SLURM_JOB_OUT

for task_id in $(seq 0 $((total_tasks - 1))); do
    # Launch the Python script with the task-specific input
    python3 -u par_get_kee_pseudowords_avg.py --task_id $task_id &
done

# Wait for all tasks to complete
wait
