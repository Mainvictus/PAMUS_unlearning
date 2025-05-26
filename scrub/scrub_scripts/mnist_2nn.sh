#!/bin/bash
echo 'Running Scrubbing Script for some Ablations...'

startID=$1
endID=$2
HessType=${3:-"Sekhari"} # 如果没有提供第三个参数，将使用 'Sekhari' 作为默认值
cuda_id=$4

dataset='mnist'
model='Logistic2NN'
epochs=50

lr=0.001
batch_size=256

orig_trainset_size=50000
used_training_size=50000
nRemovals=500
scrubType='IP'
weight_decay=0.01
order='Hessian'
FOCIType='full'
cheap_foci_thresh=0.05

approxType='FD'
n_perturbations=500

val_gap_skip=0.05

hessian_device='cpu'

delta=0.01
epsilon=0.1

# delta=0.005
# epsilon=0.05

outfile='results/mnist_2nn(Entropy_epsilon2.5_steps50).csv'

MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"
# MODEL_FILE="/root/unlearning/unlearning-attack/unlearning-attack/LCODEC-deep-unlearning/scrub/trained_models/mnist_Logistic2NN_1000_seed_0_epochs_50_lr_0.1_wd_0.01_bs_256_optim_sgd_notransform.pt"
echo "Model file path: $MODEL_FILE"

# 注释掉原始的模型训练命令，因为它目前被注释了
# CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset $dataset --model $model \
#                     --epochs $epochs --weight_decay $weight_decay \
#                     --batch_size $batch_size
# CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset $dataset --model $model \
#                     --epochs $epochs --weight_decay $weight_decay --used_training_size $used_training_size\
#                     --batch_size $batch_size --learning_rate $lr --log_loss --orig_trainset_size $orig_trainset_size

run () {
    echo "Running with HessType: $HessType"
    CUDA_VISIBLE_DEVICES=$cuda_id python multi_scrub.py --dataset $dataset \
                --model $model \
                --orig_trainset_size $orig_trainset_size \
                --used_training_size $used_training_size \
                --batch_size $batch_size \
                --train_epochs $epochs \
                --run $1 \
                --order $order \
                --selectionType $2 \
                --HessType $HessType \
                --approxType $approxType \
                --scrubType $scrubType \
                --lr $lr \
                --l2_reg $weight_decay \
                --n_perturbations $n_perturbations \
                --n_removals $nRemovals \
                --delta $delta \
                --epsilon $epsilon \
                --outfile $outfile \
                --hessian_device $hessian_device \
                --val_gap_skip $val_gap_skip
}

for runID in $(seq $startID 1 $endID)
    do
        # for selectionType in 'FOCI'
        for selectionType in 'FOCI'
            do
                run $runID $selectionType &
            done
        wait
    done
