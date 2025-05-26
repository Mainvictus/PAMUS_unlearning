#!/bin/bash
echo 'Running Scrubbing Script for some Ablations...'

cuda_id=2

dataset='mnist'
model='LogisticRegressor'
epochs=50

lr=0.1
batch_size=256

echo $MODEL_FILE

orig_trainset_size=60000
used_training_size=60000
nRemovals=500
scrubType='IP'
weight_decay=0.01
order='Hessian'
FOCIType='full'
cheap_foci_thresh=0.05

approxType='FD'
n_perturbations=500

val_gap_skip=0.05

hessian_device='cuda'

delta=0.01
epsilon=0.1

# delta=0.005
# epsilon=0.05

outfile='results/mnist_logistic(bw_combine_entropy_attack_25steps_2.5e).csv'

MODEL_FILE="trained_models/${dataset}_${model}_epochs_${epochs}.pt"
echo $MODEL_FILE

# CUDA_VISIBLE_DEVICES=$cuda_id python train.py --dataset $dataset --model $model \
#                     --epochs $epochs --weight_decay $weight_decay --used_training_size $used_training_size\
#                     --batch_size $batch_size --learning_rate $lr --log_loss --orig_trainset_size $orig_trainset_size
wait

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
                --val_gap_skip $val_gap_skip \
                --train_lr $lr \
                --train_bs $batch_size
}


for selectionType in 'FOCI' 'Full' 
    do
    for HessType in 'Sekhari'
        do
        run $selectionType $HessType &
        done
    done


