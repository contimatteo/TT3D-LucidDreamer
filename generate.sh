###

exit 1


GPU=0
PROMPT="test_t3bench_n4"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${PROMPT}"
PROMPT_DIR="${ROOT_DIR}/prompts"
PROMPT_FILE="${PROMPT_DIR}/${PROMPT}/prompts.txt"


###


CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/LucidDreamer/" \
  --train-steps=400 \
  --use-priors \
  --skip-existing
