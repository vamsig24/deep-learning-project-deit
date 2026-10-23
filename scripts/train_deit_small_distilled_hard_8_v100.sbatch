#!/bin/bash
#SBATCH --job-name=deit-s-dist-hard
#SBATCH --account=cis260364p
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:8
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# DeiT-S with hard distillation from a RegNetY-16GF teacher (README_deit.md).
# Submit from the repository root:  sbatch scripts/train_deit_small_distilled_hard_8_v100.sbatch
# Estimated ~6-17 min/epoch (the teacher forward roughly doubles the step time), so
# 100 epochs should fit in 24h; check epoch_time_min in W&B after the first epochs.
# If the job runs out of time, resubmit the same script (same RUN_NAME and EPOCHS)
# and it resumes from $OUTPUT/checkpoint.pth, e.g.
#   sbatch --dependency=afterany:<jobid> scripts/train_deit_small_distilled_hard_8_v100.sbatch

set -euo pipefail

# Avoid CPU thread oversubscription across eight training processes.
export OMP_NUM_THREADS=1

# Paths
REPO="${SLURM_SUBMIT_DIR}"
IMAGENET=/ocean/datasets/community/imagenet
SHARED=/ocean/projects/cis260364p/shared
# The cosine LR schedule spans EPOCHS, so keep it fixed for a given RUN_NAME.
EPOCHS="${EPOCHS:-100}"
RUN_NAME="${RUN_NAME:-deit-small-distilled-hard-${EPOCHS}ep-8v100}"
OUTPUT="/ocean/projects/cis260364p/$USER/deit-runs/$RUN_NAME"

# PSC does not provide the DeiT teacher; fetch it once into shared storage.
# Pre-download from a login node to avoid relying on compute-node network access.
TEACHER_URL=https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth
TEACHER_DIR="$SHARED/deit-teachers"
TEACHER="$TEACHER_DIR/regnety_160-a5fe301d.pth"
TEACHER_SHA256_PREFIX=a5fe301d

# Configure W&B. Export WANDB_API_KEY before sbatch (the job inherits your environment).
export WANDB_ENTITY="${WANDB_ENTITY:-vamsi-grandhi-carnegie-mellon-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-IDL Project}"
export WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
export WANDB_DIR="$OUTPUT"

cd "$REPO"

# Fail before staging data if required paths or the environment are unavailable.
: "${LOCAL:?PSC node-local storage path is not set}"
test -f main.py
test -r "$IMAGENET/train.tar"
test -r "$IMAGENET/val.tar"
command -v uv
: "${WANDB_API_KEY:?export WANDB_API_KEY before running sbatch}"

if [[ ! -s "$TEACHER" ]]; then
    echo "Downloading teacher weights to $TEACHER..."
    mkdir -p "$TEACHER_DIR"
    curl -fL --retry 3 -o "$TEACHER.part.$SLURM_JOB_ID" "$TEACHER_URL"
    mv -- "$TEACHER.part.$SLURM_JOB_ID" "$TEACHER"
fi
# torch.hub names checkpoints by the leading characters of their SHA-256.
sha256sum "$TEACHER" | grep -q "^$TEACHER_SHA256_PREFIX" \
    || { echo "Teacher checksum mismatch: $TEACHER" >&2; exit 1; }

# Build the environment once here so eight ranks do not race to create it.
uv sync

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

mkdir -p "$OUTPUT"

# Resume automatically when a previous job for this RUN_NAME left a checkpoint.
RESUME_ARGS=()
if [[ -s "$OUTPUT/checkpoint.pth" ]]; then
    echo "Resuming from $OUTPUT/checkpoint.pth"
    RESUME_ARGS=(--resume "$OUTPUT/checkpoint.pth")
fi

# Reuse one W&B run id per output directory so resumed jobs extend the same run.
if [[ ! -s "$OUTPUT/wandb_run_id" ]]; then
    echo "${RUN_NAME}-${SLURM_JOB_ID}" > "$OUTPUT/wandb_run_id"
fi
WANDB_RUN_ID="$(< "$OUTPUT/wandb_run_id")"
WANDB_ARGS=(--wandb --wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT"
            --wandb-name "$WANDB_NAME" --wandb-id "$WANDB_RUN_ID")

# torchrun spawns eight DDP processes and sets RANK/WORLD_SIZE/LOCAL_RANK for utils.py.
# Global batch 1024 / 8 GPUs = 128 per GPU (README uses 4 x 256); main.py scales
# lr 5e-4 by 1024/512, matching the paper. 40 CPUs give five loader workers per GPU.
# Train
uv run --no-sync torchrun --standalone --nproc_per_node=8 main.py \
    --model deit_small_distilled_patch16_224 \
    --distillation-type hard \
    --teacher-model regnety_160 \
    --teacher-path "$TEACHER" \
    --batch-size 128 \
    --epochs "$EPOCHS" \
    --data-set IMNET \
    --data-path "$LOCAL/imagenet" \
    --num_workers 5 \
    --dist-eval \
    --output_dir "$OUTPUT" \
    "${WANDB_ARGS[@]}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
