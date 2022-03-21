#!/bin/bash
#SBATCH -A e31408               # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 04:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=18G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=6     # Number of Cores (Processors)
#SBATCH --job-name="running_pmi_5"       # Name of job

module load python-anaconda3/2019.10

python bias_audit_withio.py --filter contradiction --keywords identity_labels.txt --out result_contradiction_noCA.json --dir /projects/e31408/data/a4/snli_1.0/snli_1.0_train.jsonl