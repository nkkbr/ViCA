#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1

# conda activate vica2
# cd [YOUR PROJECT FOLDER]
# export PYTHONPATH=.:$PYTHONPATH
# bash scripts/myscript_20250414_pretrain/play.sh >> scripts/myscript_20250414_pretrain/play1.log 2>&1

# export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:32

export PYTHONPATH=$(pwd):$PYTHONPATH

export PATH=$HOME/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-12.1/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=4,5,6,7
export TRITON_CACHE_DIR=/tmp/triton_autotune
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# Set up the data folder
IMAGE_FOLDER="/LLaVA-CC3M-Pretrain-595K/images"
VIDEO_FOLDER="XXX"
DATA_YAML="scripts/ViCA2/pretrain/exp1.yaml" # e.g exp.yaml

############### Prepare Envs #################
# python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

################ Arnold Jobs ################

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
#

# BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="vica2_pretrain"
# PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2"
PREV_STAGE_CHECKPOINT="./vica2_original_checkpoint"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"


# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
deepspeed --master_port 30001 \
    vica2/train/train_mem.py \
    --deepspeed scripts/zero2_offload.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts "sam_projector" \
    --mm_vision_tower_lr 1e-3 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/pretrain/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 64 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;