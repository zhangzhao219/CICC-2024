import os
import json
import torch
import random
from torch.utils.data import Dataset
from utils.rich_tqdm import progress
from utils.get_bert_and_tokenizer import getTokenizer


class CICCDataset(Dataset):
    def __init__(self, args, file_path: str, max_length: int):
        logger = args["logger"]
        tokenizer = getTokenizer(
            args["logger"],
            os.path.join(
                args["model_path"]["pretrained_model_dir"],
                args["model_path"]["tokenizer"],
            ),
        )

        if os.path.isfile(file_path) is False:
            logger.error(f"Input file path {file_path} not found")
            return
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        flatten_data = []
        for k1, v1 in data.items():
            for k2, v2 in v1.items():
                for k3, v3 in v2["Dialog"].items():
                    flatten_data.append(data[k1][k2]["Dialog"][k3])

        texts = [line["Text"] + " " + line["output"] for line in flatten_data]

        try:
            labels_str = [line["EmoAnnotation"] for line in flatten_data]
            labels = self._map_label_str_to_label(labels_str)
        except:
            logger.warning("Missing Data Label")
            labels = [-1 for line in flatten_data]

        if args["mode"] == "train" and "train" in file_path:
            type_dict = {}
            max_num = 0
            for index, text_now in enumerate(texts):
                labels_str_now = labels_str[index]
                labels_now = labels[index]
                if labels_now not in type_dict:
                    type_dict[labels_now] = []
                type_dict[labels_now].append((text_now, labels_str_now, labels_now))
                max_num = max(max_num, len(type_dict[labels_now]))

            texts = []
            labels_str = []
            labels = []
            for label, text_type_list in type_dict.items():
                if len(text_type_list) < max_num:
                    type_dict[label].extend(
                        random.choices(text_type_list, k=max_num - len(text_type_list))
                    )
                for i, j, k in type_dict[label]:
                    texts.append(i)
                    labels_str.append(j)
                    labels.append(k)

        self.dataset_len = len(texts)
        batch_encoding = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        self.data = []
        rich_dataset_id = progress.add_task(
            description="Prepare Dataset", total=self.dataset_len
        )
        if args["device"] != "CPU" and args["device"] == args["global_device"]:
            progress.start()
        for i in range(self.dataset_len):
            self.data.append(
                (
                    {
                        "input_ids": torch.tensor(
                            batch_encoding["input_ids"][i], dtype=torch.long
                        ),
                        "token_type_ids": torch.tensor(
                            batch_encoding["token_type_ids"][i], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            batch_encoding["attention_mask"][i], dtype=torch.long
                        ),
                        "labels": torch.tensor(labels[i], dtype=torch.long),
                    },
                    {"text": texts[i], "label_str": labels_str[i]},
                )
            )
            progress.advance(rich_dataset_id, advance=1)
        progress.stop()
        progress.remove_task(rich_dataset_id)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i):
        return self.data[i]

    def _map_label_str_to_label(self, label_list):
        real_label_list = []
        emotion_dict = {
            "Anger": 1,
            "Neutral": 0,
            "Happy": 2,
            "Surprise": 3,
            "Disgust": 4,
            "Sad": 5,
            "Fear": 6,
        }
        for i in label_list:
            real_label_list.append(emotion_dict[i])
        return real_label_list
