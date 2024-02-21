###

exit 1


GPU=1
ENV="test"
PROMPT="n4"
EXPERIMENT_PREFIX="t3bench/single"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"
PROMPT_FILE="${ROOT_DIR}/prompts/${EXPERIMENT_PREFIX}/${PROMPT}.txt"


export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1


###


rm -rf ./output

CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/LucidDreamer/" \
  --train-steps=850 \
  --use-priors \
  --skip-existing
