# qwen2
batch_list=(1 128 128)
dropout_list=(0.7 0.7 0.8)

sed -i "7s/.*/  train: \"data\/train_data_Qwen2-72B-Instruct_analysis.json\"/" config/CICC_textcnn.yaml
sed -i "8s/.*/  eval: \"data\/val_data_Qwen2-72B-Instruct_analysis.json\"/" config/CICC_textcnn.yaml
sed -i "9s/.*/  test: \"data\/test_data_Qwen2-72B-Instruct_analysis.json\"/" config/CICC_textcnn.yaml
sed -i "3s/.*/  train: 128/" config/CICC_textcnn.yaml
sed -i "28s/.*/dropout: 0.7/" config/CICC_textcnn.yaml
bash template.sh CICC_textcnn.yaml main_CICC_textcnn.py

# qwen2.5
batch_list=(1 4 8 16 16 16 32 64 64 64 64 64 128 128 128 256 256 256 256 256 256 256 512 512 512 512 512 512 512 1024 1024 1024 1024)
dropout_list=(0.7 0.8 0.6 0.3 0.6 0.9 0.7 0.2 0.3 0.5 0.8 0.9 0.5 0.6 0.8 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.1 0.2 0.3 0.5)

sed -i "7s/.*/  train: \"data\/train_data_Qwen2___5-72B-Instruct_analysis.json\"/" config/CICC_textcnn.yaml
sed -i "8s/.*/  eval: \"data\/val_data_Qwen2___5-72B-Instruct_analysis.json\"/" config/CICC_textcnn.yaml
sed -i "9s/.*/  test: \"data\/test_data_Qwen2___5-72B-Instruct_analysis.json\"/" config/CICC_textcnn.yaml

for i in "${!batch_list[@]}"; do
    sed -i "3s/.*/  train: ${batch_list[i]}/" config/CICC_textcnn.yaml
    sed -i "28s/.*/dropout: ${dropout_list[i]}/" config/CICC_textcnn.yaml
    bash template.sh CICC_textcnn.yaml main_CICC_textcnn.py
done
