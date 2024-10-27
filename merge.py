import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

RESULT_FOLDER = "result"
DATA_FOLDER = "data"
MODE = "val"

MERGE_FILE_LIST = [
    "2024_10_25_15_33_35",  # 0.367002014520443
    "2024_10_26_01_14_18",  # 0.34349508388742184
    "2024_10_26_04_21_48",  # 0.36280445373166537
    "2024_10_26_10_37_05",  # 0.3651463253824448
    "2024_10_26_10_37_22",  # 0.34693583319320426
    "2024_10_26_23_25_05",  # 0.38045089759885403
    "2024_10_26_23_25_11",  # 0.35723913070429025
    "2024_10_27_02_40_38",  # 0.3482362557298523
    "2024_10_27_05_26_27",  # 0.3516241233381444
    "2024_10_27_05_29_46",  # 0.34914607240230067
    "2024_10_27_05_45_03",  # 0.3489551952271762
    "2024_10_27_05_51_30",  # 0.3511076712039511
    "2024_10_27_05_57_10",  # 0.35273099636035404
    "2024_10_27_06_14_31",  # 0.34928148476510534
    "2024_10_27_06_23_00",  # 0.3500503140247399
    "2024_10_27_20_03_09",  # 0.34054465199967954
]

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
    print(result_file)
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
                result_data["label"],
                result_data["predict"].astype(int),
                output_dict=True,
            )["macro avg"]["f1-score"]
        )

    # print(
    #     classification_report(
    #         result_data["7"].map(emotion_dict), ori_data_label, output_dict=True
    #     )
    # )
# print(np.array(result_data_list).mean(axis=0))
final_result = pd.DataFrame(np.array(result_data_list).mean(axis=0))
final_result.columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
final_result["predict"] = (
    final_result[["0", "1", "2", "3", "4", "5", "6"]]
    .idxmax(axis=1)
    .astype(int)
    .map(emotion_dict)
)

if MODE == "val":
    print(
        classification_report(
            ori_data_label, final_result["predict"], output_dict=True
        )["macro avg"]["f1-score"]
    )


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
