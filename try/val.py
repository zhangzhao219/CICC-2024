import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

emotion_dict = {
    "生气": "Anger",
    "正常": "Neutral",
    "高兴": "Happy",
    "惊讶": "Surprise",
    "厌恶": "Disgust",
    "伤心": "Sad",
    "害怕": "Fear",
}

# [106353, 100416, 102483, 107664, 117577, 109384, 105187]

emotion_dict_token = {}
now_index = 0
for key, value in emotion_dict.items():
    emotion_dict_token[now_index] = value
    now_index += 1

print(emotion_dict_token)

with open("data/val_data.json", "r") as f:
    ori_data = json.load(f)

ori_data_label = []
for k1, v1 in ori_data.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2["Dialog"].items():
            ori_data_label.append(ori_data[k1][k2]["Dialog"][k3]["EmoAnnotation"])

# array_dict = {}
# global_index = 0
# for npyfile in tqdm(os.listdir("npy/val")):
#     data = np.load(os.path.join("npy/val", npyfile))
#     array_dict[str(global_index)] = data
#     global_index += 1

# np.savez('val.npz', **array_dict)

predict_list = []
predict_dict = np.load("val.npz")
for key, value in predict_dict.items():
    now_label = emotion_dict_token[np.argmax(value[0,:])]
    predict_list.append(now_label)


score = classification_report(ori_data_label, predict_list, zero_division=0)
print(score)