#!/bin/bash
#
#SBATCH --job-name=pseudowords
#SBATCH --comment="Training of pseudoword embeddings."
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tim.sockel@campus.lmu.de
#SBATCH --chdir=/home/s/sockel/Desktop/llm-x-cxg/src/pseudowords
#SBATCH --output=/home/s/sockel/Desktop/llm-x-cxg/out/slurm.%A.%a.%j.%N.out
#SBATCH --array=1-15  # Set the number of tasks as an array

# Calculate total_tasks using a subshell and Python
total_tasks=$(python -c "import itertools
import json
with open('../../out/CoMaPP_all.json') as json_file:
    data = json.load(json_file)
data.sort(key=lambda x: x['label'])
data = [list(group) for _, group in itertools.groupby(data, key=lambda x: x['label'])]
print(len(data))")

# Print total_tasks to the output file
echo "Total Tasks: $total_tasks"

python3 -u par_get_kee_pseudowords_avg.py --task_id $SLURM_ARRAY_TASK_ID

# Wait for the task to complete
wait