#!/bin/bash
## NECCESSARY JOB SPECIFICATIONS
#SBATCH --job-name=csce642-test1
#SBATCH --output=cheetah_regular_4_run.%j
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

### -------- CONFIGURATION -------- ###
ENVIRONMENT="HalfCheetah-v4"     # options: HalfCheetah-v4, Hopper-v4, Walker2d-v4
N_EPOCHS=5
NUM_CPU=1
SEED=0
DEMO_FILE=demo_data/halfcheetah_expert_demos.npz
N_CYCLES=10
N_BATCHES=40
BC_LOSS=0
Q_FILTER=0
LOG_NAME="HalfCheetah-v4_cpu1_epochs5_cycles10_0_vanilla_test"
#HalfCheetah-v4_epochs300_cpu1_cycles20_IL_QF_0_Expert

### -------- RUN PYTHON SCRIPT -------- ###
python src/experiment/train_mujoco.py \
    --env "$ENVIRONMENT" \
    --log_name "$LOG_NAME" \
    --n_epochs "$N_EPOCHS" \
    --n_cycles "$N_CYCLES" \
    --seed "$SEED" \
    --bc_loss "$BC_LOSS" \
    --q_filter "$Q_FILTER"