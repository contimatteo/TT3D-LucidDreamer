###

exit 1


# GPU=1
ENV="test"
PROMPT="n4"
EXPERIMENT_PREFIX="t3bench/single"

ROOT_DIR="/media/data2/mconti/TT3D"
OUT_DIR="${ROOT_DIR}/outputs/${ENV}/${EXPERIMENT_PREFIX}/${PROMPT}"


###


# CUDA_VISIBLE_DEVICES=${GPU} python3 tt3d_export.py \
python3 tt3d_export.py \
  --source-path "${OUT_DIR}/LucidDreamer/" \
  --skip-existing
