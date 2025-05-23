# --nnodes 1 --nproc_per_node 4 --master_port 25641


deepspeed --include localhost:0,1,2,3 train_instruct_sft_all_case.py \
    --deepspeed ds_zero2_offload.json\
    --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
    --use_lora true \
    --save_path "llama_3B_Instruct_cora_link" \
    --use_deepspeed true \
    --data_path "output/cora/link_prediction/train" \
    --bf16 false \
    --fp16 true \
    --output_dir "output_model/llama_3B_Instruct_cora_link_all_case" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --tf32 false \
    --model_max_length 2048

# --save_steps 1000 \
