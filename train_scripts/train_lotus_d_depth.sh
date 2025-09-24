export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME='/vip_media/jinshan/huggingface'
export MODELSCOPE_CACHE='/vip_media/jinshan/modelscope/hub'
export MODELSCOPE_MODULES_CACHE='/vip_media/jinshan/modelscope/modelscope_modules'
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# export PYTHONPATH="$(dirname "$(dirname "$0")"):$PYTHONPATH"

export MODEL_NAME="stabilityai/stable-diffusion-2-base"

# training dataset
export TRAIN_DATA_DIR_HYPERSIM="/vip_media/jinshan/Data/DIS5K"
# export TRAIN_DATA_DIR_VKITTI=$PATH_TO_VKITTI_DATA
export RES_HYPERSIM=576 # 所有数据会被resize成这
# export RES_VKITTI=375
export P_HYPERSIM=1.0 # 混合数据集训练时，1.0的概率会用HYPERSIM这个数据集
export NORMTYPE="trunc_disparity" # The normalization type for the depth prediction

# training configs
export BATCH_SIZE=4
export CUDA=4567
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs
export TIMESTEP=999
export TASK_NAME="depth"

# eval
export BASE_TEST_DATA_DIR="datasets/eval/"
export VALIDATION_IMAGES="datasets/quick_validation/"
export VAL_STEP=250 # 评估一次
export CHECKPOINT_STEP=500 # 保存一次模型

# output dir
export OUTPUT_DIR="output/train-lotus-d-${TASK_NAME}-bsz${TOTAL_BSZ}/"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  --main_process_port="13324" \
  train_lotus_d.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --resolution_hypersim=$RES_HYPERSIM \
  --prob_hypersim=$P_HYPERSIM \
  --mix_dataset \
  --random_flip \
  --norm_type=$NORMTYPE \
  --dataloader_num_workers=0 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20000 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --timestep=$TIMESTEP \
  --validation_images=$VALIDATION_IMAGES \
  --validation_steps=$VAL_STEP \
  --checkpointing_steps=$CHECKPOINT_STEP \
  --base_test_data_dir=$BASE_TEST_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint="latest" \
  --checkpoints_total_limit 10