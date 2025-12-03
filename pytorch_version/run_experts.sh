#!/bin/bash
## NECCESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Job1csce642
#SBATCH --output=test_1_pytorch_run_for_creating_expert_demos.%j
#SBATCH --time=00:30:00       #set for 30 minutes, should only use up 0.5 units if you leave all other settings alone which you should, run this before the other run.sh to have the demos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu        #gpu because there is no cpu partition name




### -------- MODULES + ENV -------- ###
module load GCC/11.3.0
module load CUDA/11.7.0
module load cuDNN/8.5.0
module load Python/3.8.6



source activate_venv class_project_venv



### -------- RUN PYTHON SCRIPT -------- ###
python src/utils/generate_demos.py