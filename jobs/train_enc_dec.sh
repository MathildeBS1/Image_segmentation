#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_encdec
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_encdec_%J.out
#BSUB -e outputs/output_encdec_%J.err

mkdir -p outputs

module load python3/3.13.9

# Activate uv environment
cd ~/projects/Image_segmentation
uv sync

# Print submitted parameters for debugging
echo "DATASET=${DATASET:-PH2}"
echo "LOSS=${LOSS:-dice}"

# Run training
uv run -m scripts.train_enc_dec \
    --dataset "${DATASET:-PH2}" \
    --loss "${LOSS:-dice}" \
    --epochs ${EPOCHS:-50} \
    --batch_size ${BATCH_SIZE:-4} \
    --lr ${LR:-1e-4} \
    --patience ${PATIENCE:-10} \
    --num_workers 4 \
    --device cuda \
    --checkpoint_base checkpoints