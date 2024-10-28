import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from itertools import combinations
from tqdm import tqdm

RESULT_FOLDER = "result"
DATA_FOLDER = "data"
MODE = "val"


MERGE_FILE_LIST = [
    # qwen25_textcnn
    "2024_10_27_20_29_12",  # 0.3415699355235579
    "2024_10_27_21_11_39",  # 0.34430202607376964
    "2024_10_27_21_23_05",  # 0.34598152373231933
    "2024_10_27_02_40_38",  # 0.3482362557298523
    "2024_10_27_21_29_47",  # 0.3414428016895288
    "2024_10_27_21_33_45",  # 0.34225494452270183
    "2024_10_27_21_37_43",  # 0.3418965755768518
    "2024_10_27_21_40_21",  # 0.34444451908477847
    "2024_10_27_21_42_19",  # 0.34510735380317986
    "2024_10_27_21_44_18",  # 0.34445479187435807
    "2024_10_27_21_46_16",  # 0.3477371517644962
    "2024_10_27_21_48_14",  # 0.34724691368332145
    "2024_10_27_05_26_27",  # 0.3516241233381444
    "2024_10_27_05_29_46",  # 0.34914607240230067
    "2024_10_27_21_50_12",  # 0.34011061584026325
    "2024_10_27_21_51_59",  # 0.3439476981084161
    "2024_10_27_21_53_35",  # 0.34638290126424526
    "2024_10_27_21_55_11",  # 0.3444849759434746
    "2024_10_27_05_51_30",  # 0.3511076712039511
    "2024_10_27_21_56_49",  # 0.3442332346927534
    "2024_10_27_05_57_10",  # 0.35273099636035404
    "2024_10_27_21_58_24",  # 0.3490334883620493
    "2024_10_27_22_00_00",  # 0.34527029883433386
    "2024_10_27_22_01_34",  # 0.34710509009712265
    "2024_10_27_06_14_31",  # 0.34928148476510534
    "2024_10_27_22_03_04",  # 0.34162032533104675
    "2024_10_27_22_04_43",  # 0.3474766970288084
    "2024_10_27_06_23_00",  # 0.3500503140247399
    "2024_10_27_22_06_15",  # 0.3452267161417951
    "2024_10_27_22_07_47",  # 0.3429239317458519
    "2024_10_27_22_09_20",  # 0.3452853153406492
    "2024_10_27_22_10_51",  # 0.3460207473671952
    "2024_10_27_22_12_21",  # 0.34171560905285675
    # qwen25_ernie
    "2024_10_27_12_13_17",  # 0.35323439085204933
    "2024_10_26_10_37_05",  # 0.3651463253824448
    "2024_10_25_15_33_35",  # 0.367002014520443
    "2024_10_26_23_25_05",  # 0.38045089759885403
    # qwen2textcnn
    "2024_10_26_01_14_18",  # 0.34349508388742184
    "2024_10_27_20_03_09",  # 0.34054465199967954
    "2024_10_27_05_45_03",  # 0.3489551952271762
    # qwen2_ernie
    "2024_10_27_12_13_29",  # 0.3529041491583787
    "2024_10_26_10_37_22",  # 0.34693583319320426
    "2024_10_26_23_25_11",  # 0.35723913070429025
    # yi_ernie
    "2024_10_26_04_21_48",  # 0.36280445373166537
]


def combine(temp_list):
    end_list = []
    for i in range(2, len(MERGE_FILE_LIST)):
        print(i)
        temp_list2 = []
        for c in combinations(temp_list, i):
            temp_list2.append(list(c))
        end_list.extend(temp_list2)
    return end_list


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

max_score = 0
NOW_MERGE_FILE_LIST = []

for SUB_MERGE_FILE_LIST in tqdm(combine(MERGE_FILE_LIST)):
    for result_file in SUB_MERGE_FILE_LIST:
        # print(result_file)
        result_data = pd.read_csv(
            os.path.join(RESULT_FOLDER, f"{MODE}_data_{result_file}.csv")
        )
        result_data_list.append(result_data.to_numpy())
        if MODE == "val":
            result_data["label"] = result_data["7"]
            result_data["predict"] = result_data[
                ["0", "1", "2", "3", "4", "5", "6"]
            ].idxmax(axis=1)
            # print(
            #     classification_report(
            #         result_data["label"],
            #         result_data["predict"].astype(int),
            #         output_dict=True,
            #     )["macro avg"]["f1-score"]
            # )

        # print(
        #     classification_report(
        #         result_data["7"].map(emotion_dict), ori_data_label, output_dict=True, zero_division=0
        #     )
        # )
    # print(np.array(result_data_list).mean(axis=0))
    final_result = pd.DataFrame(np.array(result_data_list).mean(axis=0))
    # final_result.columns = ["0", "1", "2", "3", "4", "5", "6", "7"]
    final_result["predict"] = (
        final_result[[0, 1, 2, 3, 4, 5, 6]].idxmax(axis=1).astype(int).map(emotion_dict)
    )

    if MODE == "val":
        score = classification_report(
            ori_data_label, final_result["predict"], output_dict=True, zero_division=0
        )["macro avg"]["f1-score"]
        if score > max_score:
            max_score = score
            NOW_MERGE_FILE_LIST = SUB_MERGE_FILE_LIST
            print(score, NOW_MERGE_FILE_LIST)


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
