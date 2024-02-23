###

exit 1


GPU=0
ENV="report"
PROMPT="n100"
EXPERIMENT_PREFIX="t3bench/single"

# ROOT_DIR="/media/data2/mconti/TT3D"
PROMPT_FILE="/media/data2/mconti/TT3D/prompts/${EXPERIMENT_PREFIX}/${PROMPT}.txt"
OUT_DIR="/media/data3/mconti/TT3D/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"


# export TRANSFORMERS_OFFLINE=1
# export DIFFUSERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1


###


rm -rf ./output


### INFO: with priors
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/LucidDreamer/" \
  --train-steps=850 \
  --use-priors \
  --skip-existing


### INFO: without priors
CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_generate.py \
  --prompt-file $PROMPT_FILE \
  --out-path "${OUT_DIR}/LucidDreamer-nopriors/" \
  --train-steps=850 \
  --skip-existing
