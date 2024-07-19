#!/bin/bash

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="FoodDialogues_test_transform_9.json"
EVAL_DATA="FoodDialogues_test_transform_7.json"
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 true \
    --fp16_full_eval true \
    --do_train \
    --do_eval \
    --tune_vision true \
    --tune_llm false \
    --use_lora true \
    --lora_target_modules "llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj)" \
    --model_max_length 2048 \
    --max_slice_nums 9 \
    --max_steps 7 \
    --eval_steps 3 \
    --output_dir output/output_minicpmv2_lora \
    --logging_dir output/output_minicpmv2_lora \
    --logging_strategy "steps" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --report_to "wandb" # wandb
# conda activate MiniCPM-V
# pip install accelerate

# 推理代码

from peft import PeftModel
from transformers import AutoModel
model_type="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
path_to_adapter="output/output_minicpmv2_lora"

model =  AutoModel.from_pretrained(
        model_type,
        trust_remote_code=True
        )

lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()

chat_model = MiniCPMVChat(lora_model)

pip install ms-swift -U
https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/minicpm-v-2.5%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html#id3
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type minicpm-v-v2_5-chat \
    --ckpt_dir output/output_minicpmv2_lora \
    --load_dataset_config true \

CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type minicpm-v-v2_5-chat \
    --ckpt_dir output/output_minicpmv2_lora \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type minicpm-v-v2_5-chat \
    --ckpt_dir output/output_minicpmv2_lora-merged \
    --load_dataset_config true
# deploy
https://swift.readthedocs.io/zh-cn/latest/Multi-Modal/MLLM%E9%83%A8%E7%BD%B2%E6%96%87%E6%A1%A3.html#minicpm-v-v2-5-chat
# 使用原始模型
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type minicpm-v-v2_5-chat

# 使用微调后的LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx

# 使用微调后Merge LoRA的模型
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model_type minicpm-v-v2_5-chat \
     --ckpt_dir output/output_minicpmv2_lora-merged

from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('cat.png', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# images = [img_base64]

# use local_path
# from swift.llm import convert_to_base64
# images = ['cat.png']
# images = convert_to_base64(images=images)['images']

# use url
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']

query = '描述这张图片'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '这张图是如何产生的？'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, history, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: minicpm-v-v2_5-chat
query: 描述这张图片
response: 这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛皮主要是白色的，带有灰色和黑色的条纹，尤其是在脸部周围。小猫的眼睛很大，呈蓝色，给人一种好奇和迷人的表情。耳朵尖尖，竖立着，显示出警觉。背景模糊不清，突出了小猫的特征。整体的色调柔和，猫咪的毛皮与背景的柔和色调形成对比。
query: 这张图是如何产生的？
response: 这张图片看起来是用数字绘画技术创作的。艺术家使用数字绘图工具来模仿毛皮的纹理和颜色，眼睛的反射，以及整体的柔和感。这种技术使艺术家能够精确地控制细节和色彩，创造出逼真的猫咪形象。
"""
