#!/bin/bash
echo 'Running Scrubbing Script for some Ablations...'

dataset='mnist'
model='LogisticRegressor'
epochs=10
cuda_id=2

lr=0.01
batch_size=256

echo $MODEL_FILE

orig_trainset_size=60000
used_training_size=60000
nRemovals=1000
scrubType='IP'
weight_decay=0.01
order='Hessian'
FOCIType='full'
cheap_foci_thresh=0.05

approxType='FD'
n_perturbations=1000

val_gap_skip=0.05

hessian_device='cuda'

delta=0.01
epsilon=0.1

outfile='results/mnist_logistic.csv'

MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"
echo $MODEL_FILE

# CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset $dataset --model $model \
#                     --epochs $epochs --weight_decay $weight_decay --orig_trainset_size $orig_trainset_size \
#                     --batch_size $batch_size --learning_rate $lr --used_training_size $used_training_size --log_loss

# wait

run () {

CUDA_VISIBLE_DEVICES=$cuda_id python multi_scrub.py --dataset $dataset \
                --model $model \
                --orig_trainset_size $orig_trainset_size \
                --used_training_size $used_training_size \
                --batch_size $batch_size \
                --train_epochs $epochs \
                --order $order \
                --selectionType $1 \
                --HessType $2 \
                --approxType $approxType \
                --scrubType $scrubType \
                --l2_reg $weight_decay \
                --n_perturbations $n_perturbations \
                --n_removals $nRemovals \
                --delta $delta \
                --epsilon $epsilon \
                --outfile $outfile \
                --hessian_device $hessian_device \
                --train_lr $lr \
                --train_bs $batch_size \
                --val_gap_skip $val_gap_skip \
                --train_lr $lr \
                --train_bs $batch_size \
                --val_gap_skip $val_gap_skip \
                --unlearning_attack white_box \
                --attack_type $3
}

for selectionType in 'FOCI'
    do
    for HessType in 'Sekhari'
        do
        for attackType in 'loss' 'max' 'max_gradnorm' 'max_diff'
            do
                run  $selectionType $HessType $attackType &
            done
        done
    done

CUDA_VISIBLE_DEVICES=3 python multi_scrub.py --dataset $dataset \
                --model $model \
                --orig_trainset_size $orig_trainset_size \
                --used_training_size $used_training_size \
                --batch_size $batch_size \
                --train_epochs $epochs \
                --order $order \
                --selectionType FOCI \
                --HessType Sekhari \
                --approxType $approxType \
                --scrubType $scrubType \
                --l2_reg $weight_decay \
                --n_perturbations $n_perturbations \
                --n_removals $nRemovals \
                --delta $delta \
                --epsilon $epsilon \
                --outfile $outfile \
                --hessian_device $hessian_device \
                --train_lr $lr \
                --train_bs $batch_size \
                --val_gap_skip $val_gap_skip \
                --train_lr $lr \
                --train_bs $batch_size \
                --val_gap_skip $val_gap_skip &
