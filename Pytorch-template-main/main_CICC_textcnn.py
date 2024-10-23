import os
import argparse
from utils.rich_tqdm import progress
from dataset.TextCNNDataset import TextCNNDataset
from model.TextCNN import TextCNN
from trainer import Trainer
from utils.config import get_config
from torch.distributed import destroy_process_group

parser = argparse.ArgumentParser(description="Pytorch NLP")
parser.add_argument("--gpu", type=str, required=True, help="Use GPU")
parser.add_argument("--datetime", type=str, required=True, help="Get Time Stamp")
parser.add_argument("--config_file", type=str, required=True, help="Config File")
args = parser.parse_args()

if __name__ == "__main__":
    # 获取配置
    args = get_config(args)
    args["logger"].info(f"Passed args {args}")

    train_dataset = TextCNNDataset(
        args=args,
        file_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["train"]
        ),
        vocab_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["vocab"]
        ),
        max_length=args["CICC"]["max_length"],
    )

    eval_dataset = TextCNNDataset(
        args=args,
        file_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["eval"]
        ),
        vocab_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["vocab"]
        ),
        max_length=args["CICC"]["max_length"],
    )

    test_dataset = TextCNNDataset(
        args=args,
        file_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["test"]
        ),
        vocab_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["vocab"]
        ),
        max_length=args["CICC"]["max_length"],
    )

    args["logger"].info(train_dataset[0])
    args["logger"].info(eval_dataset[0])

    trainer = Trainer(
        args=args,
        model=TextCNN(
            logger=args["logger"],
            dropout=args["dropout"],
            kwargs=args["CICC"],
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=None,
        predict_dataset=None,
    )

    if args["mode"] == "train":
        trainer.train()
    elif args["mode"] == "eval":
        trainer.eval()
    elif args["mode"] == "predict":
        trainer.predict()
    else:
        args["logger"].info(f"Unrecognized trainer mode: {args['mode']}")

    progress.stop()

    if args["device"] != "CPU":
        destroy_process_group()

    args["logger"].info("Program Exited Normally")
