import torch
import torch.nn as nn


class CICC_TextCNN(nn.Module):
    def __init__(self, logger, dropout, kwargs):
        super(CICC_TextCNN, self).__init__()
        self.embedding = nn.Embedding(kwargs["vocab_size"], kwargs["embedding_dim"])
        self.cnn = nn.Conv1d(
            in_channels=kwargs["embedding_dim"],
            out_channels=kwargs["num_filters"],
            kernel_size=kwargs["kernel_size"],
        )
        self.linear = nn.Sequential(
            nn.Linear(kwargs["num_filters"], kwargs["hidden_dim"]),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["num_classes"]),
        )

    def forward(self, mode, criterion, **kwargs):
        labels = kwargs["labels"]
        inputs = kwargs["input_ids"]
        embeddings = self.embedding(inputs)
        embeddings = embeddings.permute(0, 2, 1)
        cnn_output = self.cnn(embeddings)
        cnn_output = torch.max(cnn_output, dim=-1)[0]
        output = self.linear(cnn_output)

        if mode == "train":
            return criterion(output, labels)
        elif mode == "eval":
            return (
                labels.cpu().numpy().tolist(),
                output.argmax(dim=-1).cpu().numpy().tolist(),
            )
        elif mode == "predict":
            return (
                labels.cpu().numpy().tolist(),
                output.cpu().numpy(),
            )
