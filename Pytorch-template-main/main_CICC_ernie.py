import os
import argparse
from utils.rich_tqdm import progress
from dataset.CICCDataset import CICCDataset
from model.CICC_ERNIE import CICC_ERNIE
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

    train_dataset = CICCDataset(
        args=args,
        file_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["train"]
        ),
        max_length=args["CICC"]["max_length"],
    )

    eval_dataset = CICCDataset(
        args=args,
        file_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["eval"]
        ),
        max_length=args["CICC"]["max_length"],
    )

    test_dataset = CICCDataset(
        args=args,
        file_path=os.path.join(
            args["data_path"]["data_dir"], args["data_path"]["test"]
        ),
        max_length=args["CICC"]["max_length"],
    )

    args["logger"].info(train_dataset[0])
    args["logger"].info(eval_dataset[0])
    # exit()

    trainer = Trainer(
        args=args,
        model=CICC_ERNIE(
            logger=args["logger"],
            bert=os.path.join(
                args["model_path"]["pretrained_model_dir"],
                args["model_path"]["pretrained_model"],
            ),
            dropout=args["dropout"],
            num_classes=args["CICC"]["num_classes"],
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=None,
        predict_dataset=test_dataset,
    )

    if args["mode"] == "train":
        trainer.train()
    elif args["mode"] == "eval":
        trainer.eval()
    elif args["mode"] == "test":
        trainer.eval()
        trainer.predict()
    else:
        args["logger"].info(f"Unrecognized trainer mode: {args['mode']}")

    progress.stop()

    if args["device"] != "CPU":
        destroy_process_group()

    args["logger"].info("Program Exited Normally")
