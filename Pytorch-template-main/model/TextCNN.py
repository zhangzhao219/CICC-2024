import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, logger, dropout, kwargs):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(kwargs["vocab_size"], kwargs["embedding_dim"])
        self.cnn = nn.Conv1d(
            in_channels=kwargs["embedding_dim"],
            out_channels=kwargs["num_filters"],
            kernel_size=kwargs["kernel_size"],
        )
        self.linear = nn.Sequential(
            nn.Linear(
                kwargs["num_filters"], kwargs["hidden_dim"]
            ),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(kwargs["hidden_dim"], kwargs["num_classes"]),
        )

    def forward(self, mode, criterion, **kwargs):
        labels = kwargs["labels_int"]
        inputs = kwargs["text_int"]
        embeddings = self.embedding(inputs)
        embeddings = embeddings.permute(0, 2, 1)
        # print(embeddings.shape)
        cnn_output = self.cnn(embeddings)
        # print(cnn_output.shape)
        cnn_output = torch.max(cnn_output, dim=-1)[0]
        # print(cnn_output.shape)
        # exit()
        output = self.linear(cnn_output)

        if mode == "train":
            return criterion(output, labels)
        elif mode == "eval":
            return (
                labels.cpu().numpy().tolist(),
                output.argmax(dim=-1).cpu().numpy().tolist(),
            )
