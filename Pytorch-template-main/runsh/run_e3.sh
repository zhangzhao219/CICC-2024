dataset_list=("Yi-1___5-34B-Chat-16K")

dropout_list=(0.7 0.6 0.8 0.5 0.4 0.9 0.3 0.1 0.2)

for dataset in ${dataset_list[*]}; do
    echo $dataset
    sed -i "7s/.*/  train: \"data\/train_data_${dataset}_analysis.json\"/" config/CICC_ernie.yaml
    sed -i "8s/.*/  eval: \"data\/val_data_${dataset}_analysis.json\"/" config/CICC_ernie.yaml
    sed -i "9s/.*/  test: \"data\/test_data_${dataset}_analysis.json\"/" config/CICC_ernie.yaml
    for dropout in ${dropout_list[*]}; do
        echo $dropout
        sed -i "29s/.*/dropout: ${dropout}/" config/CICC_ernie.yaml
        bash template3.sh CICC_ernie.yaml main_CICC_ernie.py
    done
done