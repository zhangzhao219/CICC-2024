from copy import deepcopy
import os
import sys
import json

from swift.llm import (
    ModelType,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
)

MODEL_PATH = "/mnt/data1/model"

LLM_DICT = {
    "01ai/Yi-1___5-34B-Chat-16K": (ModelType.yi_1_5_34b_chat_16k, 4, 16384),
    "deepseek-ai/DeepSeek-V2___5": (ModelType.deepseek_v2_5, 4, 16384),
    "qwen/Qwen2___5-72B-Instruct": (ModelType.qwen2_5_72b_instruct, 8, 32768),
    "qwen/Qwen2___5-7B-Instruct": (ModelType.qwen2_5_7b_instruct, 4, 32768),
}

LLM_NAME = sys.argv[1]

model_type = LLM_DICT[LLM_NAME][0]

llm_engine = get_vllm_engine(
    model_type,
    model_id_or_path=os.path.join(MODEL_PATH, LLM_NAME),
    gpu_memory_utilization=0.95,
    tensor_parallel_size=LLM_DICT[LLM_NAME][1],
    engine_kwargs={"distributed_executor_backend": "ray"},
    max_model_len=LLM_DICT[LLM_NAME][2],
)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 512
llm_engine.generation_config.temperature = 0


def inference(llm_engine, template, data_infer):
    return inference_vllm(
        llm_engine, template, data_infer, use_tqdm=True, verbose=False
    )


DATA_FILE = sys.argv[2]

with open(os.path.join("data", DATA_FILE), "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data.keys())

PROMPT = "假如你是一个群体会话用户情感分析专家，请你对下面从中文电视剧中抽取的对话片段中两个人的每一次交流的情感进行分析。你可以标注的情感标签有7种，分别是：【高兴】【惊讶】【伤心】【生气】【厌恶】【害怕】【正常】。我会提供给你两个对话者的一些基本信息，包括姓名、年龄、性别和别名，你需要输出你认为这个对话内容属于哪个情感标签，并简要分析理由。\n示例如下：\n马得福：得宝跑了\n马得福在说得宝跑了的这件事情，没有情感倾向。因此情感标签应为【正常】\n"

HISTORY = []

# print(PROMPT)

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
# AnswerList = []

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
            # sub_answer_list.append(emotion_dict[v3["EmoAnnotation"]])
        DialogList.append(sub_dialog_list)
        # AnswerList.append(sub_answer_list)

# print(SpeakerInfoList) # ""
# print(DialogList) # []

# exit()

HistoryList = []
row_num = len(SpeakerInfoList)

for i in range(row_num):
    temp_json = {}
    temp_json["system"] = PROMPT + SpeakerInfoList[i]
    temp_json["history"] = deepcopy(HISTORY)
    temp_json["query"] = ""
    HistoryList.append(temp_json)


ResponseList = [[] for i in range(row_num)]

now_dialog = 0
while 1:
    # print(HistoryList[1])
    now_infer_data = []
    index_list = []
    for i in range(row_num):
        if len(DialogList[i]) <= now_dialog:
            continue
        HistoryList[i]["query"] = DialogList[i][now_dialog]
        now_infer_data.append(HistoryList[i])
        index_list.append(i)
    if len(now_infer_data) == 0:
        break
    response_data = inference(llm_engine, template, now_infer_data)
    for i, d in enumerate(response_data):
        ResponseList[index_list[i]].append(d["response"])
        d.pop("response")
        HistoryList[index_list[i]] = d
    now_dialog += 1


flatten_ResponseList = []

for i in range(len(ResponseList)):
    for j in range(len(ResponseList[i])):
        flatten_ResponseList.append(ResponseList[i][j])

global_index = 0
for k1, v1 in data.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2["Dialog"].items():
            data[k1][k2]["Dialog"][k3]["output"] = flatten_ResponseList[global_index]
            global_index += 1

with open(
    os.path.join(
        "data",
        DATA_FILE.replace(".json", "")
        + "_"
        + LLM_NAME.split("/")[-1]
        + "_analysis.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
