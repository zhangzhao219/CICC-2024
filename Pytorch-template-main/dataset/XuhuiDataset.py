import os
import torch
import json
from torch.utils.data import Dataset
from typing import Dict
from utils.rich_tqdm import progress
from utils.get_bert_and_tokenizer import getTokenizer


class XuhuiDataset(Dataset):
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
            data = f.readlines()
        data = [i.split("\t") for i in data]

        try:
            labels_str = [line[0].strip() for line in data]
            labels = self._map_label_str_to_label(labels_str)
        except:
            logger.warning("Missing Data Label")
            labels = [-1 for line in flatten_data]

        texts = [line[1].strip() for line in data]
        self.dataset_len = len(data)
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
