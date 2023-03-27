MODEL_SIZE=7
SEQ_LEN=256
GC_SCALE=1
DATE=20230303
echo "Training with seq_len=${SEQ_LEN} and gc_scale=${GC_SCALE}"
PER_DEVICE_BATCH_SIZE=$((256 * $GC_SCALE / $SEQ_LEN))
NUM_NODES=1
HOST_ADDR=127.0.0.1
# Hack copy it once to make it faster later
#mkdir -p ~/.checkpoints
#gsutil -m cp -r /artifacts/chatbot/${MODEL_SIZE}b/sharegpt-${DATE}-seq-${SEQ_LEN} ~/.checkpoints
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=4 \
  --master_port=12375 \
  --master_addr=$HOST_ADDR \
  --node_rank=0 \
  chatserver/train/alpaca_train.py \
  --model_name_or_path /shared/llama_weights/hf-llama-${MODEL_SIZE}b \
  --data_path /home/haozhang/test-hao/ChatServer/data/alpaca_data.json \
  --bf16 True \
  --output_dir /home/haozhang/test-hao/ChatServer/ckpts/ \
  --num_train_epochs 3 \
  --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $((128 * 512 / $SEQ_LEN / $PER_DEVICE_BATCH_SIZE / $NUM_NODES)) \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 900 \
  --save_total_limit 100 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --model_max_length ${SEQ_LEN} \
  --gradient_checkpointing True \
#  --lazy_preprocess True