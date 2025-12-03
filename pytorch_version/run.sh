#!/bin/bash
## NECCESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Job13csce642    #need to have a job name, dont have to change though becuase this is what displayed in Active Jobs (also generates .%j stuff automatically) only need to name well if have multiple and want to tell one to stop early manually or to figure out which failed early
#SBATCH --output=cheetah_regular_4_run.%j  #=cheetah_top_il_0_run.%j #the .%j lets it do the job id number keep because ensures no overwriting
#SBATCH --time=48:00:00                    #48 hours then it stops no matter what, if finishes sooner then it will also end the job, hh:mm:ss
#SBATCH --nodes=1                #leave this the same this tells it how many to use, the more the more units you use up and this code does not need more
#SBATCH --ntasks-per-node=1      #leave this the same this tells it how many to use, the more the more units you use up and this code does not need more
#SBATCH --cpus-per-task=1        #leave this the same this tells it how many to use, the more the more units you use up and this code does not need more
#SBATCH --mem=4G                 #this is how much memory it is allowed to use for the whole thing, I have seem 8G at the extreme end, this was sufficent for the tests
#SBATCH --partition=gpu        #gpu because there is no cpu partition name, see under Que Avaability in the Dashboard drop down the non legacy versoin, probably best to leave alone

### -------- CONFIGURATION -------- ###
ENVIRONMENT="HalfCheetah-v4"     # options: HalfCheetah-v4, Hopper-v4, Walker2d-v4
N_EPOCHS=200
NUM_CPU=1                       #dont change
SEED=4                          #change this!!!!!!!!!! for all 5 of the runs for the particualr experiment to be 0 to 4, it auto renames the results_log folder, you will have to on output= above yourself
POLICY_SAVE_INTERVAL=40         #this tells it how many Epochs to wait before saving a back up check point model, it starts at Epoch 0 (which is after N_CYCLES many episodes have ellapsed)
REPLAY_STRATEGY="none"          #dont change
CLIP_RETURN=1                   #dont change, I really dont recall what this does

# Leave DEMO_FILE *unset* so test -n works properly
DEMO_FILE #unset DEMO_FILE#DEMO_FILE         #this needs to be either "unset DEMO_FILE" when doing no IL or q filtering, otherwise have it be "DEMO_FILE" to do IL or Q filtering I have it set to do this right now
DEMO_LEVEL="top"                # top, mid, random     #lines 62-72 show the hard coded expert demos being used for each of these based on env and stuff
N_CYCLES=20                     #number of episodes per epoch, if we find that nothing improves the models, we could try lowering this and the POLICY_SAVE_INTERVAL so we can have the very first models which for my attempt did better in some cases
N_BATCHES=20                    #might not need to change, it worked, however, it just lets it know how many episodes to do before trying to update to be good at them so with 20 and 20 epsiodes, it tries to be best at all 20 epsiodes at same time, so maybe lowering this could = better perfromance too
BC_LOSS=0                       # 1 = do Imitation Learning (IL)
Q_FILTER=0                      # 1 = do Q filtering (I did this with BC_LOSS=1 each time as well) also be careful of spacings they have to be exact for run.sh

BASE_LOGDIR="result_logs"       #the folder it will put all the other folders in 

### -------- MODULES + ENV -------- ### dont touch these, this is how it can run on hprc properly without having to type these every single time
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

### -------- AUTO SELECT DEMO FILE IF NEEDED -------- ### This is what path gets used when it is not unset DEMO_FILE and the thing you use for DEMO_LEVEL = top, mid, or random so replace a better top file path if you figure one out
if [[ "$BC_LOSS" == 1 || "$Q_FILTER" == 1 ]]; then
    case "$ENVIRONMENT:$DEMO_LEVEL" in
        "HalfCheetah-v4:top")    DEMO_FILE="demo_data/halfcheetah_medium_high_demos.npz" ;;
        "HalfCheetah-v4:mid")    DEMO_FILE="demo_data/halfcheetah_medium_demos.npz" ;;
        "HalfCheetah-v4:random") DEMO_FILE="demo_data/halfcheetah_random_demos.npz" ;;
        "Hopper-v4:top")         DEMO_FILE="demo_data/hopper_medium_high_demos.npz" ;;
        "Hopper-v4:mid")         DEMO_FILE="demo_data/hopper_medium_demos.npz" ;;
        "Hopper-v4:random")      DEMO_FILE="demo_data/hopper_random_demos.npz" ;;
        "Walker2d-v4:top")       DEMO_FILE="demo_data/walker2d_medium_demos.npz" ;;
        "Walker2d-v4:mid")       DEMO_FILE="demo_data/walker2d_medium_low_demos.npz" ;;
        "Walker2d-v4:random")    DEMO_FILE="demo_data/walker2d_random_demos.npz" ;;
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