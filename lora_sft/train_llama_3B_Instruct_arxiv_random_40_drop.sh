# --nnodes 1 --nproc_per_node 4 --master_port 25641


deepspeed --include localhost:0,1,2,3 train_instruct_sft.py \
    --deepspeed ds_zero2_offload.json\
    --model_name_or_path "meta-llama/Llama-3.2-3B-Instruct" \
    --use_lora true \
    --save_path "llama_3B_Instruct_arxiv_node" \
    --use_deepspeed true \
    --data_path "output/random_40_drop/arxiv/node_classification/train" \
    --bf16 true \
    --fp16 false \
    --output_dir "output_model/heterophily/random_40_drop/llama_3B_Instruct_arxiv_node" \
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
    --model_max_length 1024

# --save_steps 1000 \
