#!/bin/bash
set -v
set -e
set -x

# 默认参数
NUM_SAMPLES=95  # 默认生成1个随机数
NUM_RUNS=1    # 默认运行20次
MAX_RANGE=95     # 随机数范围是1到96

if [ $# -ge 1 ]; then
    NUM_SAMPLES=$1
fi

if [ $# -ge 2 ]; then
    NUM_RUNS=$2
fi

if [ $# -ge 3 ]; then
    MAX_RANGE=$3
fi

mkdir -p ./log

# run on full dataset
for i in $(seq 1 $NUM_RUNS)
do
    RANDOM_NUMBERS=""
    LOG_FILENAME="sample_idx_1-99"
    
    for j in $(seq 0 $NUM_SAMPLES)
    do
        RANDOM_NUM=${j}
        
        RANDOM_NUMBERS="$RANDOM_NUMBERS $RANDOM_NUM"
    done
    
    echo "run times #$i: sample idx list $RANDOM_NUMBERS"
    
    python main.py --select_idxs $RANDOM_NUMBERS >> ./log/${LOG_FILENAME}.log 2>&1
    
done

# run on select dataset

NUM_SAMPLES=95  # 默认生成1个随机数
NUM_RUNS=0   # 默认运行20次
MAX_RANGE=95     # 随机数范围是1到96

for i in $(seq 1 $NUM_RUNS)
do
    RANDOM_NUMBERS=""
    LOG_FILENAME=""
    
    for j in $(seq 1 $NUM_SAMPLES)
    do
        RANDOM_NUM=$(( $RANDOM % $MAX_RANGE))
        
        RANDOM_NUMBERS="$RANDOM_NUMBERS $RANDOM_NUM"
        
        if [ -z "$LOG_FILENAME" ]; then
            LOG_FILENAME="sample_idx_${RANDOM_NUM}"
        else
            LOG_FILENAME="${LOG_FILENAME}_${RANDOM_NUM}"
        fi
    done
    
    echo "run times #$i: sample idx list $RANDOM_NUMBERS"
    
    python main.py --select_idxs $RANDOM_NUMBERS >> ./log/${LOG_FILENAME}.log 2>&1
    
done

echo "finish"