task="cola"
model="base"
seed="101"
graph_type="skeleton"

CUDA_VISIBLE_DEVICES=0 python train.py \
        --gpus 1  \
        --do_train  \
        --task $task \
        --data_dir glue_data/cola \
        --output_dir ${graph_type}_output \
        --graph_dropout 0.1 \
        --final_dropout \
        --model_name_or_path roberta-$model \
        --max_seq_length 128 \
        --num_train_epochs 1 \
        --train_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --seed $seed \
        --eval_batch_size 4 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.06 \
        --weight_decay 0.1 \
        --formalism dm \
        --n_graph_layers 2 \
        --n_graph_attn_composition_layers 2 \
        --graph_n_bases 80 \
        --graph_dim 256  \
        --post_combination_layernorm \
        --graph_type $graph_type \

