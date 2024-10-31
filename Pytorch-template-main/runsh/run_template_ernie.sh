# yi
dropout_list=(0.6 0.9)
sed -i "7s/.*/  train: \"data\/train_data_Yi-1___5-34B-Chat-16K_analysis.json\"/" config/CICC_ernie.yaml
sed -i "8s/.*/  eval: \"data\/val_data_Yi-1___5-34B-Chat-16K_analysis.json\"/" config/CICC_ernie.yaml
sed -i "9s/.*/  test: \"data\/test_data_Yi-1___5-34B-Chat-16K_analysis.json\"/" config/CICC_ernie.yaml
for dropout in ${dropout_list[*]}; do
    echo $dropout
    sed -i "29s/.*/dropout: ${dropout}/" config/CICC_ernie.yaml
    bash template.sh CICC_ernie.yaml main_CICC_ernie.py
done

# qwen2
dropout_list=(0.2 0.5 0.6 0.8 0.9)
sed -i "7s/.*/  train: \"data\/train_data_Qwen2-72B-Instruct_analysis.json\"/" config/CICC_ernie.yaml
sed -i "8s/.*/  eval: \"data\/val_data_Qwen2-72B-Instruct_analysis.json\"/" config/CICC_ernie.yaml
sed -i "9s/.*/  test: \"data\/test_data_Qwen2-72B-Instruct_analysis.json\"/" config/CICC_ernie.yaml
for dropout in ${dropout_list[*]}; do
    echo $dropout
    sed -i "29s/.*/dropout: ${dropout}/" config/CICC_ernie.yaml
    bash template.sh CICC_ernie.yaml main_CICC_ernie.py
done

# qwen2.5
dropout_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
sed -i "7s/.*/  train: \"data\/train_data_Qwen2___5-72B-Instruct_analysis.json\"/" config/CICC_ernie.yaml
sed -i "8s/.*/  eval: \"data\/val_data_Qwen2___5-72B-Instruct_analysis.json\"/" config/CICC_ernie.yaml
sed -i "9s/.*/  test: \"data\/test_data_Qwen2___5-72B-Instruct_analysis.json\"/" config/CICC_ernie.yaml
for dropout in ${dropout_list[*]}; do
    echo $dropout
    sed -i "29s/.*/dropout: ${dropout}/" config/CICC_ernie.yaml
    bash template.sh CICC_ernie.yaml main_CICC_ernie.py
done
