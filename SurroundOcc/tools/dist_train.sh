#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
SAVE_PATH=$3
#PORT=${PORT:-28108}
PORT=${PORT:-28109}
NCCL_DEBUG=INFO
#v2 12345 v3 300 v1 plus 666
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir ${SAVE_PATH} --seed 666 --launcher pytorch ${@:4} --deterministic
