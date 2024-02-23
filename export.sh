###

exit 1


# GPU=3
ENV="report"
PROMPT="n100"
EXPERIMENT_PREFIX="t3bench/single"

ROOT_DIR="/media/data3/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"


###


# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
python3 tt3d_export.py \
  --source-path "${OUT_DIR}/LucidDreamer/" \
  --skip-existing


python3 tt3d_export.py \
  --source-path "${OUT_DIR}/LucidDreamer-nopriors/" \
  --skip-existing
