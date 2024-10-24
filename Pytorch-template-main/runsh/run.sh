# bash template.sh CICC_textcnn.yaml main_CICC_textcnn.py

dataset_list=("Qwen2___5-72B-Instruct" "Qwen2-72B-Instruct" "Yi-1___5-34B-Chat-16K")
batch_list=(1 2 4 8 16 32 64 128 256 512 1024)
dropout_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for dataset in ${dataset_list[*]}; do
    echo $dataset
    sed -i "7s/.*/  train: \"data\/train_data_${dataset}_analysis.json\"/" config/CICC_textcnn.yaml
    for batch in ${batch_list[*]}; do
        echo $batch
        sed -i "3s/.*/  train: ${batch}/" config/CICC_textcnn.yaml
        for dropout in ${dropout_list[*]}; do
            echo $dropout
            sed -i "28s/.*/dropout: ${dropout}/" config/CICC_textcnn.yaml
            bash template.sh CICC_textcnn.yaml main_CICC_textcnn.py
        done
    done
done
