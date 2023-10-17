#!/bin/bash
#The related information about multi-nodes cluster.
MASTER_PORT="23456"
GPUS_PER_NODE="$SM_NUM_GPUS"

if [ "$bf16" = "true" ] || [ "$bf16" = "True" ]; then
    bf16_bool=true
else
    bf16_bool=false
fi

if [ "$tf32" = "true" ] || [ "$tf32" = "True" ]; then
    tf32_bool=true
else
    tf32_bool=false
fi

# Now you can use $bf16_bool in your script as a boolean.



# Now you can use $bf16_bool in your script as a boolean.


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
             --master_port $MASTER_PORT"
           

OPTS=""
OPTS+=" --model_name_or_path ${model_name_or_path}"
OPTS+=" --data_path ${data_path}"
OPTS+=" --bf16 ${bf16_bool}"
OPTS+=" --output_dir ${output_dir}"
OPTS+=" --num_train_epochs ${num_train_epochs}"
OPTS+=" --per_device_train_batch_size ${per_device_train_batch_size}"
OPTS+=" --per_device_eval_batch_size ${per_device_eval_batch_size}"
OPTS+=" --gradient_accumulation_steps ${gradient_accumulation_steps}"
OPTS+=" --evaluation_strategy ${evaluation_strategy}"
OPTS+=" --save_strategy ${save_strategy}"
OPTS+=" --save_steps ${save_steps}"
OPTS+=" --save_total_limit ${save_total_limit}"
OPTS+=" --learning_rate ${learning_rate}"
OPTS+=" --weight_decay ${weight_decay}"
OPTS+=" --warmup_ratio ${warmup_ratio}"
OPTS+=" --lr_scheduler_type ${lr_scheduler_type}"
OPTS+=" --logging_steps ${logging_steps}"
# OPTS+=' --fsdp \"full_shard auto_wrap\"'
OPTS+=" --fsdp_transformer_layer_cls_to_wrap ${fsdp_transformer_layer_cls_to_wrap}"
OPTS+=" --tf32 False"
OPTS+=" --deepspeed ${deepspeed}"


CMD="torchrun ${DISTRIBUTED_ARGS} train.py ${OPTS}"

echo ${CMD}

${CMD} 2>&1 | tee /opt/ml/output/train_log