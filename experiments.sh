#!/bin/bash

########## FINE-TUNE

# Data Substitution
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline baseline --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Joint Incremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Incremental/Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline incremental_decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_substitute.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done



########## FEATURE-EXTRACTION

# Data Substitution
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline baseline --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn yes --freeze_backbone yes --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Joint Incremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn yes --freeze_backbone yes --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn yes --freeze_backbone yes --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Incremental/Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec --pipeline incremental_decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn yes --freeze_backbone yes --early_stopping_val 10 --weight_decay 5e-4 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_substitute.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

########## EWC

# Data Substitution
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_ewc --pipeline baseline --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --ewc_lambda 500 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Joint Incremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_ewc --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --ewc_lambda 500 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_ewc --pipeline decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --ewc_lambda 500 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Incremental/Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_ewc --pipeline incremental_decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --ewc_lambda 500 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_substitute.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done


########## FD

# Data Substitution
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_fd --pipeline baseline --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --fd_lamb 0.1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Joint Incremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_fd --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --fd_lamb 0.1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_fd --pipeline decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --fd_lamb 0.1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Incremental/Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_fd --pipeline incremental_decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --fd_lamb 0.1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_substitute.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done


########## LWF

# Data Substitution
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_lwf --pipeline baseline --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --lwf_lamb 1 --lwf_T 1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Joint Incremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_lwf --pipeline joint_incremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --lwf_lamb 1 --lwf_T 1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path None --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_lwf --pipeline decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --lwf_lamb 1 --lwf_T 1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_remove.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done

# Incremental/Decremental
for i in 0 1 2
do
    python -u ./data_incdec_framework/cl_framework/main.py \
    -op path_to_output/ \
        --approach incdec_lwf --pipeline incremental_decremental --n_accumulation 4 --seed $i --nw 4 \
        --freeze_bn no --freeze_backbone no --early_stopping_val 10 --weight_decay 5e-4 \
        --lwf_lamb 1 --lwf_T 1 \
        --stop_first_task no \
        --epochs 100 --batch_size 4 --lr_first_task 1e-4 --lr_first_task_head 1e-4 --head_lr 1e-4 --backbone_lr 1e-4 \
        --scheduler_type fixd --plateau_check map --patience 10 --device 0 \
        --criterion_type multilabel --dataset kinetics --data_path ./Kinetics \
        --subcategories_csv_path Kinetics/Info/subcategories_to_substitute.csv --subcategories_randomize yes \
        --n_task 6 --sampler imbalanced \
        --backbone movinetA0 --pretrained_path ./cleaned_checkpoint_sgd.pt
done