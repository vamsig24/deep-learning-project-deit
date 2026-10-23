#!/bin/bash
# DeiT-S hard distillation on ImageNet-1k, one PSC Bridges-2 GPU node.
#
# Resource layout, ImageNet archives, node-local staging, and the uv launcher
# follow scripts/train_vit_small_8_v100.sbatch in
# https://github.com/vamsig24/deep-learning-project
# That script does not load a module and does not activate conda. It checks
# that `uv` is on PATH and runs the training command with `uv run`.
#
# Training command is the hard-distillation recipe in README_deit.md, with
# DeiT-S in place of DeiT-B:
#   --model deit_small_distilled_patch16_224
#   --distillation-type hard
#   --teacher-model regnety_160
#   --teacher-path https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
# The distilled variant is required. deit_small_patch16_224 has no distillation
# token, and losses.py rejects hard distillation unless the model returns one.
# README_deit.md launches that recipe with torch.distributed.launch --use_env.
# torchrun sets the same RANK / WORLD_SIZE / LOCAL_RANK variables.
#
# Submit from this repository root on a PSC login node:
#   sbatch scripts/train_deit_small_hard_distill_psc.sbatch
#
#SBATCH --job-name=deit-s-hard
#SBATCH --account=cis260364p
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:8
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Avoid CPU thread oversubscription across eight training processes.
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Checkout that contains main.py. Submit from the repository root, or set REPO.
REPO="${REPO:-${SLURM_SUBMIT_DIR}}"

# Same community ImageNet-1k archives as the ViT-Small 8x V100 job.
IMAGENET=/ocean/datasets/community/imagenet
OUTPUT="${OUTPUT:-/ocean/projects/cis260364p/${USER}/deit-small-hard-distill-${SLURM_JOB_ID}}"
TEACHER_DIR="/ocean/projects/cis260364p/${USER}/deit-teacher"
TEACHER_PATH="${TEACHER_DIR}/regnety_160-a5fe301d.pth"
TEACHER_URL="https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth"

cd "$REPO"
test -f "$REPO/main.py"
test -f "$REPO/pyproject.toml"

# Fail before staging data if required paths or the environment are unavailable.
: "${LOCAL:?PSC node-local storage path is not set}"
test -r "$IMAGENET/train.tar"
test -r "$IMAGENET/val.tar"
command -v uv

# Stage archives, then extract locally to avoid shared-filesystem small-file reads.
# Archives already contain train/<class>/ and val/<class>/ directories.
mkdir -p "$LOCAL/imagenet"

echo "Copying ImageNet..."
copy_started=$SECONDS
for split in train val; do
    echo "Copying ImageNet $split.tar..."
    split_started=$SECONDS
    rsync -a --info=progress2 --stats --outbuf=L \
        "$IMAGENET/$split.tar" "$LOCAL/imagenet/"
    echo "ImageNet $split archive copy complete in $((SECONDS - split_started)) seconds."
    echo "Extracting ImageNet $split (progress every approximately 1 GB)..."
    extract_started=$SECONDS
    tar --extract --file="$LOCAL/imagenet/$split.tar" \
        --directory="$LOCAL/imagenet" --no-same-owner --no-same-permissions \
        --checkpoint=100000 --checkpoint-action=echo
    test -d "$LOCAL/imagenet/$split"
    echo "ImageNet $split extraction complete in $((SECONDS - extract_started)) seconds."
    rm -- "$LOCAL/imagenet/$split.tar"
done

echo "ImageNet staging complete in $((SECONDS - copy_started)) seconds total."

mkdir -p "$OUTPUT" "$TEACHER_DIR"

# README_deit.md: download the teacher before launch so workers do not write
# the same file. Eight torchrun ranks would each hit torch.hub otherwise.
if [[ ! -s "$TEACHER_PATH" ]]; then
    echo "Downloading RegNetY-160 teacher to ${TEACHER_PATH}"
    curl -fL --retry 3 --retry-delay 5 -o "${TEACHER_PATH}.partial" "$TEACHER_URL"
    mv "${TEACHER_PATH}.partial" "$TEACHER_PATH"
fi

# One parent spawns eight DDP processes; do not launch eight parents.
# batch-size is per GPU. 128 x 8 = 1024, the DeiT global batch, so the
# built-in linear LR scale (5e-4 * global_batch / 512) stays at 1e-3.
# 40 CPUs / 8 GPUs is five data-loader workers per rank.
# --dist-eval shards the 50k-image val set across the eight ranks.
RESUME_ARGS=()
if [[ -f "${OUTPUT}/checkpoint.pth" ]]; then
    echo "Resuming from ${OUTPUT}/checkpoint.pth"
    RESUME_ARGS=(--resume "${OUTPUT}/checkpoint.pth")
fi

uv run torchrun --standalone --nproc_per_node=8 main.py \
    --model deit_small_distilled_patch16_224 \
    --distillation-type hard \
    --teacher-model regnety_160 \
    --teacher-path "$TEACHER_PATH" \
    --batch-size 128 \
    --epochs 300 \
    --data-path "$LOCAL/imagenet" \
    --data-set IMNET \
    --num_workers 5 \
    --output_dir "$OUTPUT" \
    --dist-eval \
    "${RESUME_ARGS[@]}"
