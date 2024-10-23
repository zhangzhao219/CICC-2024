import re
import os
import sys
import json
import jieba
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE_LIST = [
    "train_data_Qwen2-72B-Instruct_reason.json",
    "train_data_glm-4-9b-chat_reason.json",
    "train_data_internlm2_5-7b-chat_reason.json",
]

data_list = []

for DATA_FILE in DATA_FILE_LIST:
    with open(os.path.join("result", DATA_FILE), "r", encoding="utf-8") as f:
        data = json.load(f)
        data_list.append(data)

emotion_dict = {
    "Anger": "【生气】",
    "Neutral": "【正常】",
    "Happy": "【高兴】",
    "Surprise": "【惊讶】",
    "Disgust": "【厌恶】",
    "Sad": "【伤心】",
    "Fear": "【害怕】",
}

MAX_NUM_ALL = 240000

emotion_dict_count_maxnum = {
    "【生气】": MAX_NUM_ALL,
    "【正常】": 15000,
    "【高兴】": MAX_NUM_ALL,
    "【惊讶】": MAX_NUM_ALL,
    "【厌恶】": MAX_NUM_ALL,
    "【伤心】": MAX_NUM_ALL,
    "【害怕】": MAX_NUM_ALL,
}

emotion_dict_count_nownum = {
    "【生气】": 0,
    "【正常】": 0,
    "【高兴】": 0,
    "【惊讶】": 0,
    "【厌恶】": 0,
    "【伤心】": 0,
    "【害怕】": 0,
}

emotion_list = ["生气", "害怕", "伤心", "厌恶", "惊讶", "高兴", "正常"]

emotion_dict_count = {}

for i, data in enumerate(data_list):
    if i == 0:
        for k1, v1 in data.items():
            for k2, v2 in v1.items():
                Dialog_valid_dict = {}
                for k3, v3 in v2["Dialog"].items():
                    emo = emotion_dict[v3["EmoAnnotation"]]
                    emotion_dict_count_nownum[emo] += 1
                    if emotion_dict_count_nownum[emo] <= emotion_dict_count_maxnum[emo]:
                        Dialog_valid_dict[k3] = v3
                v2["Dialog"] = Dialog_valid_dict
    else:
        for k1, v1 in data.items():
            for k2, v2 in v1.items():
                Dialog_valid_dict = {}
                for k3, v3 in v2["Dialog"].items():
                    emo = emotion_dict[v3["EmoAnnotation"]]
                    emotion_dict_count_nownum[emo] += 1
                    if emotion_dict_count_nownum[emo] <= emotion_dict_count_maxnum[emo]:
                        data_list[0][k1][k2]["Dialog"][k3 + f"_{i}"] = v3

final_data = data_list[0]
emotion_dict_count = {}
for k1, v1 in final_data.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2["Dialog"].items():
            emo = emotion_dict[v3["EmoAnnotation"]]
            if emo not in emotion_dict_count:
                emotion_dict_count[emo] = 1
            else:
                emotion_dict_count[emo] += 1
print(emotion_dict_count)

with open(
    os.path.join("result", "train_data_15000.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)