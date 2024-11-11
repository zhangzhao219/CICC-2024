import os
import sys
import json
import numpy as np

from swift.llm import (
    ModelType,
    get_model_tokenizer,
    get_default_template_type,
    get_template,
    inference,
)

from swift.utils import seed_everything

from sklearn.metrics import classification_report

from tqdm import tqdm

MODEL_PATH = "/mnt/data1/model"

LLM_DICT = {
    "qwen/Qwen2___5-72B-Instruct": ModelType.qwen2_5_72b_instruct,
    "qwen/Qwen2-7B-Instruct": ModelType.qwen2_7b_instruct,
    "qwen/Qwen2-1___5B-Instruct": ModelType.qwen2_1_5b_instruct,
}

LLM_NAME = sys.argv[1]

model_type = LLM_DICT[LLM_NAME]

template_type = get_default_template_type(model_type)

DATA_FILE = sys.argv[2]

with open(os.path.join("data", DATA_FILE), "r", encoding="utf-8") as f:
    data = json.load(f)

SYSTEM = "假如你是一个群体会话用户情感分析专家，请你对下面从中文电视剧中抽取的对话片段中两个人的每一次交流的情感进行分析。你可以标注的情感标签有7种，分别是：【高兴】【惊讶】【伤心】【生气】【厌恶】【害怕】【正常】。我会提供给你两个对话者的一些基本信息，包括姓名、年龄、性别和别名，你需要输出这个对话内容属于哪个情感标签，不要输出任何其他的无关内容\n"

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
    "生气": "Anger",
    "正常": "Neutral",
    "高兴": "Happy",
    "惊讶": "Surprise",
    "厌恶": "Disgust",
    "伤心": "Sad",
    "害怕": "Fear",
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
        # sub_answer_list = []
        for k3, v3 in v2["Dialog"].items():
            sub_dialog_list.append(f"{Speaker_dict[v3['Speaker']]}：{v3['Text']}\n")
            AnswerList.append(v3["EmoAnnotation"])
        DialogList.append(sub_dialog_list)

kwargs = {}
kwargs["use_flash_attn"] = True
model, tokenizer = get_model_tokenizer(
    model_type,
    model_id_or_path=os.path.join(MODEL_PATH, LLM_NAME),
    model_kwargs={"device_map": "auto"},
    **kwargs,
)

emotion_dict_token = {}
extract_list = []
now_index = 0
for key, value in emotion_dict.items():
    token = tokenizer(key)["input_ids"][0]
    extract_list.append(token)
    emotion_dict_token[now_index] = value
    now_index += 1

print(extract_list)
print(emotion_dict_token)

# 修改max_new_tokens
model.generation_config.max_new_tokens = 8
model.generation_config.output_logits = True
model.generation_config.return_dict_in_generate = True

template = get_template(template_type, tokenizer)
seed_everything(42)

save_dict = {}

flatten_ResponseList = []
global_index = 0
for i in tqdm(range(len(SpeakerInfoList))):
    NOW_SYSTEM_PROMPT = SYSTEM + SpeakerInfoList[i]
    history = [
        [
            "马得福：得宝跑了",
            "【正常】",
        ]
    ]
    for j in tqdm(range(len(DialogList[i]))):
        return_dict = inference(
            model, template, DialogList[i][j], history, NOW_SYSTEM_PROMPT
        )
        history = return_dict["history"]

        all_logits = return_dict["logits"][1][:, extract_list].cpu()
        save_dict[str(global_index)] = all_logits.numpy()
        global_index += 1
        now_logits = np.argmax(all_logits.sum(axis=0), axis=-1).item()
        flatten_ResponseList.append(emotion_dict_token[now_logits])

global_index = 0
for k1, v1 in data.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2["Dialog"].items():
            data[k1][k2]["Dialog"][k3]["output"] = flatten_ResponseList[global_index]
            global_index += 1

with open(
    os.path.join(
        "data_logits",
        DATA_FILE.replace(".json", "") + "_" + LLM_NAME.split("/")[-1] + "_logits.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

score = classification_report(AnswerList, flatten_ResponseList, zero_division=0)
print(score)

np.save(f"val_n1.npz", save_dict)
