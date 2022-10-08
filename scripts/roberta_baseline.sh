task="rte"
lr="2e-5"
model="large"
seed="52"

CUDA_VISIBLE_DEVICES=0 python roberta_baseline.py \
  --model_name_or_path roberta-$model \
  --task_name $task \
  --do_train \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate $lr \
  --num_train_epochs 20 \
  --logging_steps 100 \
  --seed $seed \
  --evaluation_strategy "epoch" \
  --report_to wandb \
  --save_strategy no \
  --run_name $task-$model-$lr-$seed-wotf \
  --output_dir glue_data/rte \
  --overwrite_output
