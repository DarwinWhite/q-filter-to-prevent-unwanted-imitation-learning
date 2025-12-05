#!/bin/bash
## NECCESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Job13csce642thebatch20one
#SBATCH --output=t3_cheetah_top_q_0_runbatch20.%j #t2_cheetah_regular_4_run.%j#=t2_cheetah_top_il_0_run.%j
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G


### -------- CONFIGURATION -------- ###
ENVIRONMENT="HalfCheetah-v4"     # options: HalfCheetah-v4, Hopper-v4, Walker2d-v4
N_EPOCHS=300
NUM_CPU=1
SEED=0
POLICY_SAVE_INTERVAL=1
REPLAY_STRATEGY="none"
CLIP_RETURN=1

# Leave DEMO_FILE *unset* so test -n works properly
DEMO_FILE #unset DEMO_FILE
DEMO_LEVEL="top"                # top, mid, random
N_CYCLES=20
N_BATCHES=20 #test to see if this is how long many steps per train epsiode, it was 20 earlier
BC_LOSS=1
Q_FILTER=1

BASE_LOGDIR="result_logs"

### -------- MODULES + ENV -------- ###
module load GCC/11.3.0
module load CUDA/11.7.0
module load cuDNN/8.5.0
module load Python/3.8.6



source activate_venv class_project_venv

### -------- CREATE SUFFIX -------- ###
SUFFIX=""

if [[ "$BC_LOSS" == 1 && "$Q_FILTER" == 1 ]]; then
    SUFFIX="_IL_Q_filter_${DEMO_LEVEL}_lvl_expert"
elif [[ "$BC_LOSS" == 1 ]]; then
    SUFFIX="_IL_${DEMO_LEVEL}_lvl_expert"
elif [[ "$Q_FILTER" == 1 ]]; then
    SUFFIX="_Q_filter_${DEMO_LEVEL}_lvl_expert"
else
    SUFFIX="_regular"
fi

### -------- BUILD LOG DIRECTORY -------- ###
CYCLES_STR="${N_CYCLES:-default}"
BATCHES_STR="${N_BATCHES:-default}"

LOGDIR="${BASE_LOGDIR}/${ENVIRONMENT}_epochs${N_EPOCHS}_cpu${NUM_CPU}_cycles${CYCLES_STR}_batches${BATCHES_STR}_seed${SEED}${SUFFIX}"
mkdir -p "$LOGDIR"

### -------- AUTO SELECT DEMO FILE IF NEEDED -------- ###
if [[ "$BC_LOSS" == 1 || "$Q_FILTER" == 1 ]]; then
    case "$ENVIRONMENT:$DEMO_LEVEL" in
        "HalfCheetah-v4:top")    DEMO_FILE="demo_data/halfcheetah_expert_demos.npz" ;;
        "HalfCheetah-v4:mid")    DEMO_FILE="demo_data/halfcheetah_medium_demos.npz" ;;
        "HalfCheetah-v4:random") DEMO_FILE="demo_data_old/halfcheetah_random_demos.npz" ;;
        "Hopper-v4:top")         DEMO_FILE="demo_data/hopper_expert_demos.npz" ;;
        "Hopper-v4:mid")         DEMO_FILE="demo_data/hopper_medium_demos.npz" ;;
        "Hopper-v4:random")      DEMO_FILE="demo_data_old/hopper_random_demos.npz" ;;
        "Walker2d-v4:top")       DEMO_FILE="demo_data/walker2d_expert_demos.npz" ;;
        "Walker2d-v4:mid")       DEMO_FILE="demo_data/walker2d_medium_demos.npz" ;;
        "Walker2d-v4:random")    DEMO_FILE="demo_data_old/walker2d_random_demos.npz" ;;
    esac
fi

### -------- BUILD EXTRA ARGS SAFELY -------- ###
EXTRA_ARGS=()

[[ -n "$DEMO_FILE" ]] && EXTRA_ARGS+=("--demo_file" "$DEMO_FILE")
[[ -n "$N_CYCLES"  ]] && EXTRA_ARGS+=("--n_cycles" "$N_CYCLES")
[[ -n "$N_BATCHES" ]] && EXTRA_ARGS+=("--n_batches" "$N_BATCHES")

### -------- RUN PYTHON SCRIPT -------- ###
python src/experiment/train_mujoco.py \
    --env "$ENVIRONMENT" \
    --logdir "$LOGDIR" \
    --n_epochs "$N_EPOCHS" \
    --num_cpu "$NUM_CPU" \
    --seed "$SEED" \
    --policy_save_interval "$POLICY_SAVE_INTERVAL" \
    --replay_strategy "$REPLAY_STRATEGY" \
    --clip_return "$CLIP_RETURN" \
    "${EXTRA_ARGS[@]}" \
    --bc_loss "$BC_LOSS" \
    --q_filter "$Q_FILTER"
