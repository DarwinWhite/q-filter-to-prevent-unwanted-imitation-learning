#!/bin/bash
## NECCESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Job1csce642
#SBATCH --output=new_expert_test_4_demos_new_run.%j
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G





### -------- MODULES + ENV -------- ###
module load GCC/11.3.0
module load CUDA/11.7.0
module load cuDNN/8.5.0
module load Python/3.8.6



source activate_venv class_project_venv



### -------- RUN PYTHON SCRIPT -------- ###
# # python src/utils/generate_demos_np_policies.py
# python src/utils/generate_demos_new.py params/cheetah_params.pkl --n_episodes 10 --out demo_data/TLhalfcheetah_expert_demos.npz

### -------- RUN PYTHON SCRIPT MULTIPLE TIMES -------- ###

declare -a inputs=(
    "params/cheetah_params.pkl"
    "params/cheetah_medium_params.pkl"
    "params/walker2d_params.pkl"
    "params/walker2d_medium_params.pkl"
    "params/hopper_params.pkl"
    "params/hopper_medium_params.pkl"
)

declare -a outputs=(
    "demo_data/halfcheetah_expert_demos.npz"
    "demo_data/halfcheetah_medium_demos.npz"
    "demo_data/walker2d_expert_demos.npz"
    "demo_data/walker2d_medium_demos.npz"
    "demo_data/hopper_expert_demos.npz"
    "demo_data/hopper_medium_demos.npz"
)

declare -a tgenvs=(
    "HalfCheetah-v4"
    "HalfCheetah-v4"
    "Walker2d-v4"
    "Walker2d-v4"
    "Hopper-v4"
    "Hopper-v4"
)


for i in ${!inputs[@]}; do
    echo "Running: ${inputs[$i]} --> ${outputs[$i]}"
    python src/utils/generate_demos_new.py \
        "${inputs[$i]}" \
        --env "${tgenvs[$i]}" \
        --n_episodes 10 \
        --max_steps 1000 \
        --out "${outputs[$i]}"
done
