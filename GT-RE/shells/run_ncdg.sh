python -u train2.py \
       --display_name REGT_NCDG2 \
        --dataset chemdisgene \
        --transformer_type bert \
        --model_name_or_path ../PLM/BiomedBERT-large-uncased-abstract \
        --train_file train.json \
        --dev_file valid.json \
        --test_file test.anno_all.json \
        --save_path ../trained_model/model_REGT_NCDG.pth \
        --num_train_epochs 2.0 \
        --train_batch_size 4 \
        --test_batch_size 4 \
        --seed 66 \
        --num_class 15 \
        --tau 0.2 \
        --gpu 0 \
        --evaluation_steps 500 \
        --learning_rate 1e-5 \
        --loss SPU \