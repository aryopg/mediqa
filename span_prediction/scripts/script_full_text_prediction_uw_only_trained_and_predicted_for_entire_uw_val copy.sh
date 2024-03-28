# bash span_prediction/scripts/script_full_text_prediction_uw_only_trained_and_predicted_for_entire_uw_val.sh
num_train_epochs=30
echo "num_train_epochs = $num_train_epochs"

batch_size=32
echo "batch_size = $batch_size"

datadir=./span_prediction/data
datadir_error_only=./span_prediction/data/error_only #../data/qa/$task
echo "datadir = $datadir"

for lr in 5e-06 # 5e-06 
do
echo "lr = $lr"

for seed in 5
do
echo "seed = $seed"


# SQUAD
outdir=~/rds/hpc-work/mediqa_output/prediction/full_text/uw_val_4_aryo # ./span_prediction/output/large/full_text # squad_trained/$dataset/num_epochs_$num_train_epochs/lr_$lr/seed_$seed/
# PAQ
# outdir = 

echo "outdir = $outdir"

# fp16 cancelled
# --test_file $datadir_error_only/test_set/test_set_processed_full_text_as_input.json \

mkdir -p $outdir
python3 -u span_prediction/run_qa_mediqa.py --model_name_or_path /home/co-chae/rds/hpc-work/mediqa_output/prediction/full_text/uw_only_training/checkpoint-72 \
    --seed $seed \
    --do_predict \
    --per_device_train_batch_size $batch_size --learning_rate $lr \
    --num_train_epochs $num_train_epochs \
    --test_file /home/co-chae/mediqa/span_prediction/data/aryo_ms_val/uw_val_processed_full_text_as_input.json \
    --preprocessing_num_workers 10 \
    --max_seq_length 512 --doc_stride 128 \
    --gradient_accumulation_steps 1 \
    --save_strategy no --save_total_limit 1 --evaluation_strategy no --eval_steps 1000 --load_best_model_at_end True --metric_for_best_model f1 --greater_is_better True \
    --output_dir $outdir --overwrite_output_dir true --overwrite_cache \
    |& tee $outdir/log.txt &

done
done