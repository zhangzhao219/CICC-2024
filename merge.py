import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

RESULT_FOLDER = "result"
DATA_FOLDER = "data"
MODE = "test"

MERGE_FILE_LIST = ["2024_10_25_15_33_35"]

emotion_dict = {
    0: "Neutral",
    1: "Anger",
    2: "Happy",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Fear",
}

with open(os.path.join(DATA_FOLDER, f"{MODE}_data.json"), "r") as f:
    ori_data = json.load(f)

if MODE == "val":
    ori_data_label = []
    for k1, v1 in ori_data.items():
        for k2, v2 in v1.items():
            for k3, v3 in v2["Dialog"].items():
                ori_data_label.append(ori_data[k1][k2]["Dialog"][k3]["EmoAnnotation"])

result_data_list = []

for result_file in MERGE_FILE_LIST:
    result_data = pd.read_csv(
        os.path.join(RESULT_FOLDER, f"{MODE}_data_{result_file}.csv")
    )
    result_data_list.append(result_data.to_numpy())
    if MODE == "val":
        result_data["label"] = result_data["7"]
        result_data["predict"] = result_data[
            ["0", "1", "2", "3", "4", "5", "6"]
        ].idxmax(axis=1)
        print(
            classification_report(
                result_data["label"], result_data["predict"].astype(int)
            )
        )

    # print(
    #     classification_report(
    #         result_data["7"].map(emotion_dict), ori_data_label, output_dict=True
    #     )
    # )

final_result = pd.DataFrame(np.array(result_data_list).mean(axis=0))
final_result["predict"] = (
    result_data[["0", "1", "2", "3", "4", "5", "6"]]
    .idxmax(axis=1)
    .astype(int)
    .map(emotion_dict)
)

if MODE == "val":
    print(classification_report(ori_data_label, final_result["predict"]))


if MODE == "test":
    predict_labels = final_result["predict"].tolist()
    global_index = 0
    for k1, v1 in ori_data.items():
        for k2, v2 in v1.items():
            for k3, v3 in v2["Dialog"].items():
                ori_data[k1][k2]["Dialog"][k3]["EmoAnnotation"] = predict_labels[
                    global_index
                ]
                global_index += 1
    with open(os.path.join(RESULT_FOLDER, f"test_label.json"), "w") as f:
        json.dump(ori_data, f, ensure_ascii=False, indent=4)
