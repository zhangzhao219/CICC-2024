import os
import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from itertools import combinations
from tqdm import tqdm

RESULT_FOLDER = "result"
DATA_FOLDER = "data"
MODE = "val"

now_best_list = ['2024_10_27_21_51_59', '2024_10_27_05_51_30', '2024_10_27_22_00_00', '2024_10_27_22_04_43', '2024_10_30_11_32_21', '2024_10_29_09_56_33', '2024_10_26_23_25_05', '2024_10_26_01_14_18', '2024_10_30_11_42_32', '2024_10_27_12_13_29', '2024_10_28_07_30_06']

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
    "2024_10_29_22_44_35",  # 0.3494056503888177
    "2024_10_30_11_32_21",  # 0.34619010075218226
    "2024_10_29_09_56_33",  # 0.35236661262063596
    "2024_10_28_08_20_46",  # 0.36719007229339046
    "2024_10_27_12_13_17",  # 0.35323439085204933
    "2024_10_26_10_37_05",  # 0.3651463253824448
    "2024_10_25_15_33_35",  # 0.367002014520443
    "2024_10_26_23_25_05",  # 0.38045089759885403
    "2024_10_28_21_08_36",  # 0.37277270123503375
    # qwen2textcnn
    "2024_10_26_01_14_18",  # 0.34349508388742184
    "2024_10_27_20_03_09",  # 0.34054465199967954
    "2024_10_27_05_45_03",  # 0.3489551952271762
    # qwen2_ernie
    "2024_10_30_11_42_32",  # 0.3400349265290517
    "2024_10_27_12_13_29",  # 0.3529041491583787
    "2024_10_26_10_37_22",  # 0.34693583319320426
    "2024_10_26_23_25_11",  # 0.35723913070429025
    "2024_10_28_21_17_39",  # 0.3477689605241899
    # yi_ernie
    "2024_10_26_04_21_48",  # 0.36280445373166537
    "2024_10_28_07_30_06",  # 0.34965055869741535
]

# MERGE_FILE_LIST = [i for i in MERGE_FILE_LIST if i in now_best_list]
# print(len(MERGE_FILE_LIST))


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

for index, file in enumerate(MERGE_FILE_LIST):
    result_data_list.append(
        pd.read_csv(os.path.join(RESULT_FOLDER, f"{MODE}_data_{file}.csv"))[
            ["0", "1", "2", "3", "4", "5", "6"]
        ].to_numpy()
    )

result_data_list = np.array(result_data_list)

TIMES = 1000000
random.seed(0)
def combine(temp_list):
    end_list = []
    for i in tqdm(range(10, len(temp_list)+1)):
        for _ in tqdm(range(TIMES)):
            a = random.choices([j for j in range(len(temp_list))], k=i)
            now_list = [False for j in range(len(temp_list))]
            for c2 in a:
                now_list[c2] = True
            end_list.append(np.array(now_list))
    return end_list


# def combine(temp_list):
#     end_list = []
#     for i in tqdm(range(2, len(temp_list) + 1)):
#         a = combinations([j for j in range(len(temp_list))], i)
#         for c in tqdm(list(a)):
#             now_list = [False for j in range(len(temp_list))]
#             for c2 in c:
#                 now_list[c2] = True
#             end_list.append(np.array(now_list))
#     return end_list


SUB_MERGE_FILE_LIST = combine(MERGE_FILE_LIST)

random.seed(0)
random.shuffle(SUB_MERGE_FILE_LIST)

max_score = 0
NOW_MERGE_FILE_LIST = []

for zero_one_select in tqdm(SUB_MERGE_FILE_LIST):
    now_file_list = result_data_list[zero_one_select]

    # if MODE == "val":
    #     for i in range(now_file_list.shape[0]):
    #         ids = np.argmax(now_file_list[i], axis=1)
    #         ids_str = np.array(list(map(emotion_dict.get, ids)))
    #         print(
    #             classification_report(
    #                 ori_data_label, ids_str, output_dict=True, zero_division=0
    #             )["macro avg"]["f1-score"]
    #         )

    now_file_list = now_file_list.sum(axis=0)
    ids = np.argmax(now_file_list, axis=1)
    ids_str = np.array(list(map(emotion_dict.get, ids)))

    if MODE == "val":
        score = classification_report(
            ori_data_label, ids_str, output_dict=True, zero_division=0
        )["macro avg"]["f1-score"]
        if score > max_score:
            max_score = score
            NOW_MERGE_FILE_LIST = []
            for i, j in enumerate(zero_one_select):
                if j == 1:
                    NOW_MERGE_FILE_LIST.append(MERGE_FILE_LIST[i])
            print(score, NOW_MERGE_FILE_LIST)


if MODE == "test":
    predict_labels = ids_str.tolist()
    global_index = 0
    for k1, v1 in ori_data.items():
        for k2, v2 in v1.items():
            for k3, v3 in v2["Dialog"].items():
                ori_data[k1][k2]["Dialog"][k3]["EmoAnnotation"] = predict_labels[
                    global_index
                ]
                global_index += 1
    with open("test_label.json", "w") as f:
        json.dump(ori_data, f, ensure_ascii=False, indent=4)
