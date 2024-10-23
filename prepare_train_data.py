import os
import sys
import json


DATA_FILE = sys.argv[1]

with open(os.path.join("data", DATA_FILE), "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data.keys())

PROMPT = "假如你是一个群体会话用户情感分析专家，请你对下面从中文电视剧中抽取的对话片段中两个人的每一次交流的情感进行标注。我会提供给你两个对话者的一些基本信息，包括姓名、年龄、性别和别名，并提供每一次的对话内容，你需要对每一次的对话内容的情感进行标注。你可以标注的情感标签有7种，分别是：【高兴】【惊讶】【伤心】【生气】【厌恶】【害怕】【正常】。\n请你仅选择你认为最为贴切的标签进行输出，不允许输出其他的标签。\n"

age_dict = {
    "child": "童年",
    "mid": "中年",
    "young": "青年",
    "old": "老年",
}

gender_dict = {
    "female": "女",
    "male": "男",
}

emotion_dict = {
    "Anger": "【生气】",
    "Neutral": "【正常】",
    "Happy": "【高兴】",
    "Surprise": "【惊讶】",
    "Disgust": "【厌恶】",
    "Sad": "【伤心】",
    "Fear": "【害怕】",
}

SpeakerInfoList = []
DialogList = []
AnswerList = []

for k1, v1 in data.items():
    for k2, v2 in v1.items():
        Name_A = v2["SpeakerInfo"]["A"]["Name"]
        Age_A = age_dict[v2["SpeakerInfo"]["A"]["Age"]]
        Gender_A = gender_dict[v2["SpeakerInfo"]["A"]["Gender"]]
        OtherName_A = "，".join(v2["SpeakerInfo"]["A"]["OtherName"])
        SpeakerInfo = (
            f"对话人A基本信息：姓名：{Name_A}；年龄：{Age_A}；性别：{Gender_A}"
        )
        if OtherName_A != "":
            SpeakerInfo += f"；别名：{OtherName_A}\n"
        else:
            SpeakerInfo += "\n"

        Name_B = v2["SpeakerInfo"]["B"]["Name"]
        Age_B = age_dict[v2["SpeakerInfo"]["B"]["Age"]]
        Gender_B = gender_dict[v2["SpeakerInfo"]["B"]["Gender"]]
        OtherName_B = "，".join(v2["SpeakerInfo"]["B"]["OtherName"])
        SpeakerInfo += (
            f"对话人B基本信息：姓名：{Name_B}；年龄：{Age_B}；性别：{Gender_B}"
        )
        if OtherName_B != "":
            SpeakerInfo += f"；别名：{OtherName_B}\n"
        else:
            SpeakerInfo += "\n"

        SpeakerInfoList.append(SpeakerInfo)

        Speaker_dict = {
            "A": Name_A,
            "B": Name_B,
        }

        sub_dialog_list = []
        sub_answer_list = []
        for k3, v3 in v2["Dialog"].items():
            sub_dialog_list.append(f"{Speaker_dict[v3['Speaker']]}：{v3['Text']}\n")
            sub_answer_list.append(emotion_dict[v3["EmoAnnotation"]])
        DialogList.append(sub_dialog_list)
        AnswerList.append(sub_answer_list)

# print(SpeakerInfoList)  # ""
# print(DialogList)  # []
# print(AnswerList)  # []


row_num = len(SpeakerInfoList)

final_data_list = []

for i in range(row_num):
    temp_json = {}
    temp_json["system"] = PROMPT + SpeakerInfoList[i]
    now_dialog = DialogList[i]
    now_answer = AnswerList[i]
    history_list = []
    for j, dialog in enumerate(now_dialog):
        if j == len(now_dialog) - 1:
            temp_json["query"] = dialog
            temp_json["response"] = now_answer[j]
        else:
            history_list.append([dialog, now_answer[j]])
    temp_json["history"] = history_list
    final_data_list.append(temp_json)

with open(
    os.path.join("data", DATA_FILE.replace(".json", "") + "_model.jsonl"),
    "w",
    encoding="utf-8",
) as f:
    for d in final_data_list:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
