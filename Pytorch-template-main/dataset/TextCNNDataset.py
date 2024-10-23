import os
import string
import torch
import jieba
from torch.utils.data import Dataset
from typing import Dict
from utils.rich_tqdm import progress
from utils.get_bert_and_tokenizer import getTokenizer


class TextCNNDataset(Dataset):
    def __init__(self, args, file_path: str, vocab_path: str, max_length: int):
        logger = args["logger"]
        tokenizer = getTokenizer(
            args["logger"],
            os.path.join(
                args["model_path"]["pretrained_model_dir"],
                args["model_path"]["tokenizer"],
            ),
        )

        # if os.path.isfile(file_path) is False:
        #     logger.error(f"Input file path {file_path} not found")
        #     return
        # logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = f.readlines()

        self.dataset_len = len(data)
        self.data = []

        labels_str_list = []
        text_str_list = []
        text_int_list = []

        # vocab_dict = {}
        # if os.path.isfile(vocab_path) is True:
        # with open(vocab_path, "r", encoding="utf-8") as f:
        #     vocab_data = f.readlines()
        # for i, v in enumerate(vocab_data):
        #     vocab_dict[v.strip()] = i
        # else:
        #     logger.info(f"No vocab file, Now build one with train dataset")
        #     vocab_dict["<PAD>"] = 0

        for d in data:
            d = d.split("\t")
            labels_str_list.append(d[0].strip())
            text_str_list.append(d[1].strip())
        rich_dataset_id = progress.add_task(
            description="Prepare Dataset", total=self.dataset_len
        )
        if args["device"] != "CPU" and args["device"] == args["global_device"]:
            progress.start()

        batch_encoding = tokenizer(
            text_str_list,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )


        for i in range(self.dataset_len):
            self.data.append(
                (
                    {
                        "text_int": torch.tensor(batch_encoding["input_ids"][i], dtype=torch.long),
                        "labels_int": torch.tensor(
                            self._map_label_str_to_label(labels_str_list[i]),
                            dtype=torch.long,
                        ),
                    },
                    {"text_str": text_str_list[i], "label_str": labels_str_list[i]},
                )
            )

        # for d in data:
        #     d = d.split("\t")
        #     now_labels_str = d[0].strip()
        #     labels_str_list.append(now_labels_str)
        #     now_text_str_list = d[1].strip()
        #     now_text_int_list = [
        #         vocab_dict[x] for x in now_text_str_list if x in vocab_dict
        #     ]

        #     if len(now_text_int_list) > max_length:
        #         now_text_int_list = now_text_int_list[:max_length]
        #     else:
        #         now_text_int_list += [
        #             0 for i in range(max_length - len(now_text_int_list))
        #         ]
        #     text_str_list.append(now_text_str_list)
        #     text_int_list.append(now_text_int_list)

        #     self.data.append(
        #         (
        #             {
        #                 "text_int": torch.tensor(now_text_int_list, dtype=torch.long),
        #                 "labels_int": torch.tensor(
        #                     self._map_label_str_to_label(now_labels_str),
        #                     dtype=torch.long,
        #                 ),
        #             },
        #             {"text_str": now_text_str_list, "label_str": now_labels_str},
        #         )
        #     )
            progress.advance(rich_dataset_id, advance=1)
        progress.stop()
        progress.remove_task(rich_dataset_id)

        # if os.path.isfile(vocab_path) is False:
        #     with open(vocab_path, "w", encoding="utf-8") as f:
        #         for v, i in vocab_dict.items():
        #             f.write(str(v) + "\n")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, i):
        return self.data[i]

    def _map_label_str_to_label(self, label):
        emotion_dict = {
            "Anger": 1,
            "Neutral": 0,
            "Happy": 2,
            "Surprise": 3,
            "Disgust": 4,
            "Sad": 5,
            "Fear": 6,
        }
        return emotion_dict[label]

    def _clean_str_list(self, text_list):
        new_text_list = []
        for i in text_list:
            if i in ["】", "【", " "] or i in string.punctuation:
                continue
            new_text_list.append(i)
        return new_text_list
