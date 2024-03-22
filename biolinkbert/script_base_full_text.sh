
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=7


num_train_epochs=30
echo "num_train_epochs = $num_train_epochs"

batch_size=128
echo "batch_size = $batch_size"

datadir=./biolinkbert/data #../data/qa/$task
echo "datadir = $datadir"

for lr in 5e-06 # 5e-06 
do
echo "lr = $lr"

for seed in 5
do
echo "seed = $seed"


# SQUAD
outdir=./biolinkbert/output/base/full_text # squad_trained/$dataset/num_epochs_$num_train_epochs/lr_$lr/seed_$seed/
# PAQ
# outdir = 

echo "outdir = $outdir"

# fp16 cancelled

mkdir -p $outdir
python3 -u biolinkbert/run_qa_mediqa.py --model_name_or_path michiyasunaga/BioLinkBERT-base \
    --seed $seed \
    --do_train --do_eval --do_predict\
    --per_device_train_batch_size $batch_size --learning_rate $lr \
    --num_train_epochs $num_train_epochs \
    --train_file $datadir/ms_train_processed_full_text_as_input.json --validation_file $datadir/ms_val_processed_full_text_as_input.json --test_file $datadir/ms_val_processed_full_text_as_input.json \
    --preprocessing_num_workers 10 \
    --max_seq_length 512 --doc_stride 128 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch --save_total_limit 1 --evaluation_strategy epoch --eval_steps 1000 --load_best_model_at_end True --metric_for_best_model f1 --greater_is_better True \
    --output_dir $outdir --overwrite_output_dir true --overwrite_cache \
    |& tee $outdir/log.txt &

done
done