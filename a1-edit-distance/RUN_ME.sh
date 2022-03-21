echo "❗️Numpy is needed! so I'm loading the conda environment..."
module purge all
module load python-anaconda3/2019.10
source activate /projects/e31408/users/zhf3975/conda/LingEnv
echo "✅Done loading the env called LingEnv. Now running the test for a1."
python /projects/e31408/autograders/a1.py