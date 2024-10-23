import torch
import torch.nn as nn

from utils.get_bert_and_tokenizer import getBert


class CICC(nn.Module):
    def __init__(self, logger, bert, dropout, num_classes):
        super(CICC, self).__init__()
        self.bert = getBert(logger, bert)
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Sequential(
            nn.Linear(1024, 256), self.dropout, nn.Tanh(), nn.Linear(256, num_classes)
        )

    def forward(self, mode, criterion, **kwargs):
        input_ids = kwargs["input_ids"]
        token_type_ids = kwargs["token_type_ids"]
        attention_mask = kwargs["attention_mask"]
        labels = kwargs["labels"]
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        output = torch.mean(outputs["last_hidden_state"], 1)
        output = self.linear(output)

        if mode == "train":
            return criterion(output, labels)
        elif mode == "eval":
            return (
                labels.cpu().numpy().tolist(),
                output.argmax(dim=-1).cpu().numpy().tolist(),
            )
