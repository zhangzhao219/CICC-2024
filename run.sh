# python infer_zero_shot_reason.py deepseek-ai/DeepSeek-V2___5 test_data.json
# python infer_zero_shot_reason.py deepseek-ai/DeepSeek-V2___5 val_data.json
# python infer_zero_shot_reason.py deepseek-ai/DeepSeek-V2___5 train_data.json

# python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct test_data.json
# python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct val_data.json



# CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py Shanghai_AI_Laboratory/internlm2_5-20b-chat test_data.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py Shanghai_AI_Laboratory/internlm2_5-20b-chat val_data.json 
CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py Shanghai_AI_Laboratory/internlm2_5-20b-chat train_data.json 

# sleep 1h
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json
python infer_zero_shot_reason.py qwen/Qwen2___5-72B-Instruct train_data.json

# CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py 01ai/Yi-1___5-34B-Chat-16K test_data.json
# CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py 01ai/Yi-1___5-34B-Chat-16K val_data.json 
# CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py 01ai/Yi-1___5-34B-Chat-16K train_data.json 

# CUDA_VISIBLE_DEVICES=4,5,6,7 python infer_zero_shot_reason.py Shanghai_AI_Laboratory/internlm2_5-20b-chat train_data.json 

# CUDA_VISIBLE_DEVICES=1 python infer_zero_shot_reason.py qwen/Qwen2___5-7B-Instruct test_data.json
# CUDA_VISIBLE_DEVICES=1 python infer_zero_shot_reason.py glm-4-9b-chat val_data.json
# CUDA_VISIBLE_DEVICES=2 python infer_zero_shot_reason.py glm-4-9b-chat train_data.json

# CUDA_VISIBLE_DEVICES=2 python infer_zero_shot_reason.py internlm2_5-7b-chat test_data.json
# CUDA_VISIBLE_DEVICES=3 python infer_zero_shot_reason.py internlm2_5-7b-chat val_data.json
# CUDA_VISIBLE_DEVICES=3 python infer_zero_shot_reason.py internlm2_5-7b-chat train_data.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_zero_shot.py Qwen2-72B-Instruct test_data.json
# CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_zero_shot.py Qwen2-72B-Instruct val_data.json
# CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_zero_shot.py Qwen2-72B-Instruct train_data.json

# python infer_zero_shot.py Qwen2-7B-Instruct test_data.json
# python infer_zero_shot.py Qwen2-7B-Instruct val_data.json
# python infer_zero_shot.py Qwen2-7B-Instruct train_data.json

# python infer_zero_shot.py glm-4-9b-chat test_data.json
# python infer_zero_shot.py glm-4-9b-chat val_data.json
# python infer_zero_shot.py glm-4-9b-chat train_data.json

# python infer_zero_shot.py internlm2_5-7b-chat test_data.json
# python infer_zero_shot.py internlm2_5-7b-chat val_data.json
# python infer_zero_shot.py internlm2_5-7b-chat train_data.json

# python transform.py train_data_glm-4-9b-chat.json 2>&1 | tee -a log
# python transform.py train_data_Qwen2-72B-Instruct.json 2>&1 | tee -a log
# python transform.py train_data_Qwen2-7B-Instruct.json 2>&1 | tee -a log
# python transform.py train_data_internlm2_5-7b-chat.json 2>&1 | tee -a log

# python transform.py val_data_glm-4-9b-chat.json 2>&1 | tee -a log
# python transform.py val_data_Qwen2-72B-Instruct.json 2>&1 | tee -a log
# python transform.py val_data_Qwen2-7B-Instruct.json 2>&1 | tee -a log
# python transform.py val_data_internlm2_5-7b-chat.json 2>&1 | tee -a log

# python transform.py test_data_glm-4-9b-chat.json 2>&1 | tee -a log
# python transform.py test_data_Qwen2-72B-Instruct.json 2>&1 | tee -a log
# python transform.py test_data_Qwen2-7B-Instruct.json 2>&1 | tee -a log
# python transform.py test_data_internlm2_5-7b-chat.json 2>&1 | tee -a log


# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-100-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-200-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-300-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-400-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-500-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-600-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-700-merged val_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-776-merged val_data.json


# python transform.py val_data_checkpoint-100-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-200-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-300-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-400-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-500-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-600-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-700-merged.json 2>&1 | tee -a log
# python transform.py val_data_checkpoint-776-merged.json 2>&1 | tee -a log

# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-600-merged train_data.json
# CUDA_VISIBLE_DEVICES=0 python infer.py output/internlm2_5-7b-chat/v0-20240712-094651/checkpoint-700-merged train_data.json

# python transform.py train_data_checkpoint-600-merged.json 2>&1 | tee -a log
# python transform.py train_data_checkpoint-700-merged.json 2>&1 | tee -a log
