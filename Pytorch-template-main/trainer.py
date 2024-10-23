import os
import json
import torch
from torch.optim import AdamW
from utils.rich_tqdm import progress
from utils.metrics import calculate_metrics
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_gather_object, barrier
from transformers import get_scheduler
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union
from collections import deque


class Trainer:
    def __init__(
        self,
        args=None,
        model=None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
    ):
        self.args = args
        self.logger = args["logger"]
        self.mode = args["mode"]

        # 当前模型的最好的评价指标
        self.best_metric = {
            "train": {
                "metric": 0,
                "epoch": 0,
            },
            "eval": {
                "metric": 0,
                "epoch": 0,
            },
            "test": {
                "metric": 0,
                "epoch": 0,
            },
        }

        # 起始epoch
        self.epoch_start = 1

        # snapshot
        self.snapshot_path = os.path.join(
            self.args["model_path"]["snapshot_dir"], self.args["datetime"]
        )
        self.snapshot = None

        # 校验传递的模型
        if model is None:
            self.logger.error("Trainer requires a model argument")
            return

        if self.args["device"] != "CPU":
            self.gpu_id = int(os.environ["LOCAL_RANK"])

            self.snapshot = self._load_snapshot()

            if self.snapshot is not None:
                self.epoch_start = self.snapshot["EPOCHS_RUN"] + 1
                model.load_state_dict(self.snapshot["MODEL_STATE"])
            # find_unused_parameters没看懂什么意思
            self.model = DistributedDataParallel(
                model.to(self.gpu_id), find_unused_parameters=True
            )
        else:
            self.gpu_id = None

            # 模型存储与加载位置
            self.snapshot = self._load_snapshot()

            if self.snapshot is not None:
                self.epoch_start = self.snapshot["EPOCHS_RUN"] + 1
                model.load_state_dict(self.snapshot["MODEL_STATE"])

            self.model = model

        # 校验训练数据集
        if train_dataset is None and self.mode == "train":
            self.logger.error("No Train Dataset Passed")
            return
        self.train_dataset = train_dataset

        # 校验验证数据集
        if eval_dataset is None and self.mode == "eval":
            self.logger.error("No Eval Dataset Passed")
            return
        self.eval_dataset = eval_dataset

        # 校验测试数据集
        if test_dataset is None and self.mode == "test":
            self.logger.error("No Test Dataset Passed")
            return
        self.test_dataset = test_dataset

        # 校验预测的数据集
        if predict_dataset is None and self.mode == "predict":
            self.logger.error("No Predict Dataset Passed")
            return
        self.predict_dataset = predict_dataset

        (
            self.criterion,
            self.optimizer,
            self.scheduler,
            self.tensorboard_writer,
            self.early_stop_sign,
        ) = (None, None, None, None, None)

        if self.mode == "train":
            # 优化器与损失函数
            self.criterion = self._get_criterion()
            self.optimizer, self.scheduler = self._get_optimizer_and_scheduler()

            if self.snapshot is not None:
                self.optimizer.load_state_dict(self.snapshot["OPTIMIZER_STATE"])
                self.scheduler.load_state_dict(self.snapshot["SCHEDULER_STATE"])

            self.tensorboard_writer = SummaryWriter(
                os.path.join("runs", args["datetime"], args["device"])
            )

            self.early_stop_sign = deque(maxlen=args["early_stop"])

    # 获取当前模型的优化器
    def _get_optimizer_and_scheduler(self):
        if self.args["optimizer"] == "AdamW":
            optimizer = self._modified_optimizer()
        else:
            self.logger.error("Unrecognized Optimizer")
            exit()
        num_training_steps = (
            self.train_dataset.__len__()
            // (self.args["batch_size"]["train"] * self.args["world_size"])
        ) * self.args["max_epochs"]
        self.logger.info(f"Total train steps: {num_training_steps}")
        scheduler = get_scheduler(
            self.args["scheduler"]["name"],
            optimizer,
            num_training_steps * self.args["scheduler"]["ratio"],
            num_training_steps,
        )
        return optimizer, scheduler

    # 对optimizer进行调整
    def _modified_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.args["lr"])

    # 获取当前模型的损失函数
    def _get_criterion(self):
        if self.args["criterion"] == "CrossEntropyLoss":
            from loss.CELoss import CELoss

            return CELoss()
        elif self.args["criterion"] == "DualLoss":
            from loss.DualLoss import DualLoss

            return DualLoss(0.5, 0.1)
        elif self.args["criterion"] == "ContrastiveLoss":
            from loss.ContrastiveLoss import ContrastiveLoss

            return ContrastiveLoss()
        elif self.args["criterion"] == "MCContrastiveLoss":
            from loss.MCContrastiveLoss import MCContrastiveLoss

            return MCContrastiveLoss(
                self.train_dataset.__len__()
                // (self.args["batch_size"]["train"] * self.args["world_size"]),
                self.args["seed"],
                self.args["Stance"]["bert_config"],
            )

        elif self.args["criterion"] == "FocalLoss":
            from loss.FocalLoss import FocalLoss

            return FocalLoss(0.05, 2, 0.05, 2)

    # 准备DataLoader
    def _prepare_dataloader(self, dataset: Dataset, batch_size: int):
        if self.gpu_id is not None:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                sampler=DistributedSampler(dataset=dataset, seed=self.args["seed"]),
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

    # 训练主函数
    def train(self):
        # 计算epoch的进度条
        rich_epoch_id = progress.add_task(
            description="Epoch", total=self.args["max_epochs"]
        )
        # 创建train_dataset、eval_dataset的进度条和dataloader
        if self.train_dataset is None or self.eval_dataset is None:
            self.logger.error("No Train or Eval Dataset Passed")
            return

        self.train_dataloader = self._prepare_dataloader(
            self.train_dataset, self.args["batch_size"]["train"]
        )
        rich_train_step_id = progress.add_task(
            description="Train Step", total=len(self.train_dataloader)
        )

        self.train_eval_dataloader = self._prepare_dataloader(
            self.train_dataset, self.args["batch_size"]["eval"]
        )
        rich_train_eval_step_id = progress.add_task(
            description="Train Eval Step", total=len(self.train_eval_dataloader)
        )

        self.eval_dataloader = self._prepare_dataloader(
            self.eval_dataset, self.args["batch_size"]["eval"]
        )
        rich_eval_step_id = progress.add_task(
            description="Eval Step", total=len(self.eval_dataloader)
        )

        if self.test_dataset is not None:
            self.test_dataloader = self._prepare_dataloader(
                self.test_dataset, self.args["batch_size"]["eval"]
            )
            rich_test_step_id = progress.add_task(
                description="Test Step", total=len(self.test_dataloader)
            )

        # 启动终端进度条
        if self._judge_main_process():
            progress.start()

        # 循环训练
        for epoch in range(1, self.args["max_epochs"] + 1):
            # 控制训练的随机性
            if epoch < self.epoch_start:
                progress.update(rich_epoch_id, advance=1)
                continue

            # 首先训练
            loss = self._train_one_epoch(
                dataloader=self.train_dataloader,
                task_id=rich_train_step_id,
                epoch=epoch,
            )
            self.logger.info(f"Train Epoch: {epoch} Loss: {loss}")
            torch.cuda.empty_cache()
            # 评估模型在训练集上面的效果
            metrics_train = self._eval_one_epoch(
                dataloader=self.train_eval_dataloader, task_id=rich_train_eval_step_id
            )
            self.logger.info(f"Train Eval Epoch: {epoch} Metrics: {metrics_train}")
            if self._judge_main_process():
                if (
                    metrics_train[self.args["main_metric"]]
                    > self.best_metric["train"]["metric"]
                ):
                    self.best_metric["train"]["metric"] = metrics_train[
                        self.args["main_metric"]
                    ]
                    self.best_metric["train"]["epoch"] = epoch
                    progress.update(
                        task_id=rich_train_eval_step_id, best=self.best_metric["train"]
                    )
            # 评估模型在验证集上面的效果
            metrics_eval = self._eval_one_epoch(
                dataloader=self.eval_dataloader, task_id=rich_eval_step_id
            )
            self.logger.info(f"Eval Epoch: {epoch} Metrics: {metrics_eval}")

            # 评估模型在测试集上面的效果
            if self.test_dataset is not None:
                metrics_test = self._eval_one_epoch(
                    dataloader=self.test_dataloader, task_id=rich_test_step_id
                )
                self.logger.info(f"Test Epoch: {epoch} Metrics: {metrics_test}")
                if self._judge_main_process():
                    if (
                        metrics_test[self.args["main_metric"]]
                        > self.best_metric["test"]["metric"]
                    ):
                        self.best_metric["test"]["metric"] = metrics_test[
                            self.args["main_metric"]
                        ]
                        self.best_metric["test"]["epoch"] = epoch
                        progress.update(
                            task_id=rich_test_step_id, best=self.best_metric["test"]
                        )
            else:
                metrics_test = {}

            barrier()

            self._write_tensorboard(
                epoch, loss, metrics_train, metrics_eval, metrics_test
            )
            if self._judge_main_process():
                if self._judge_whether_save(rich_eval_step_id, epoch, metrics_eval):
                    return

            # 更新进度条
            progress.update(rich_epoch_id, advance=1)

        self.logger.info(f"Best Metric: {self.best_metric}")

    # 评估主函数
    def eval(self):
        # self.train_eval_dataloader = self._prepare_dataloader(
        #     self.train_dataset, self.args["batch_size"]["eval"]
        # )
        # rich_train_eval_step_id = progress.add_task(
        #     description="Train Eval Step", total=len(self.train_eval_dataloader)
        # )

        # self.eval_dataloader = self._prepare_dataloader(
        #     self.eval_dataset, self.args["batch_size"]["eval"]
        # )
        # rich_eval_step_id = progress.add_task(
        #     description="Eval Step", total=len(self.eval_dataloader)
        # )

        if self.test_dataset is not None:
            self.test_dataloader = self._prepare_dataloader(
                self.test_dataset, self.args["batch_size"]["eval"]
            )
            rich_test_step_id = progress.add_task(
                description="Test Step", total=len(self.test_dataloader)
            )

        # 如果有eval_dataset，在eval_dataset上面实时评估效果
        # 启动终端进度条
        if self._judge_main_process():
            progress.start()

        # # 评估模型在训练集上面的效果
        # metrics = self._eval_one_epoch(
        #     dataloader=self.train_eval_dataloader, task_id=rich_train_eval_step_id
        # )
        # self.logger.info(f"Train Eval Metrics: {metrics}")
        # # 评估模型在验证集上面的效果
        # metrics = self._eval_one_epoch(
        #     dataloader=self.eval_dataloader, task_id=rich_eval_step_id
        # )
        # self.logger.info(f"Eval Metrics: {metrics}")
        # 评估模型在测试集上面的效果
        if self.test_dataset is not None:
            metrics_test = self._eval_one_epoch(
                dataloader=self.test_dataloader, task_id=rich_test_step_id
            )
            self.logger.info(f"Test Metrics: {metrics_test}")

    # 预测主函数
    def predict(self):
        self.predict_dataloader = self._prepare_dataloader(
            self.predict_dataset, self.args["batch_size"]["predict"]
        )
        rich_predict_step_id = progress.add_task(
            description="Predict Step", total=len(self.predict_dataloader)
        )

        # 启动终端进度条
        if self._judge_main_process():
            progress.start()

        # 给测试集打标签
        data_predict = self._predict_one_epoch(
            dataloader=self.predict_dataloader, task_id=rich_predict_step_id
        )
        with open(
            os.path.join(
                self.args["data_path"]["data_dir"],
                self.args["data_path"]["predict_save_path"],
            ),
            "w",
        ) as f:
            json.dump(
                {"keywords": "test NLI task", "example": data_predict},
                f,
                ensure_ascii=False,
                indent=2,
            )
        self.logger.info("Predict Finished")

    # 训练一轮辅助函数
    def _train_one_epoch(self, task_id, dataloader, epoch):
        self.model.train()

        progress.reset(task_id)

        loss_list = []
        if self.args["device"] != "CPU":
            dataloader.sampler.set_epoch(epoch)
        # 遍历每一个step的数据，返回损失值进行统计
        for _, train_data in enumerate(dataloader):
            loss_single = self._one_batch(train_data, "train")
            loss_list.append(loss_single)
            progress.update(task_id=task_id, advance=1, loss=loss_single)
        return sum(loss_list) / len(loss_list)

    # 测试一轮辅助函数
    def _eval_one_epoch(self, task_id, dataloader):
        self.model.eval()

        progress.reset(task_id)

        self.all_eval_result = []
        temp_eval_result = []

        # 遍历每一个step的数据，返回预测值与真实值的标签进行统计
        for _, eval_data in enumerate(dataloader):
            _, eval_result = self._one_batch(eval_data, "eval")

            while len(self.all_eval_result) != len(eval_result):
                self.all_eval_result.append(
                    [[] for i in range(self.args["world_size"])]
                )
            while len(temp_eval_result) != len(eval_result):
                temp_eval_result.append([])
            for i, eval_single_result in enumerate(eval_result):
                temp_eval_result[i].extend(eval_single_result)
                all_gather_object(self.all_eval_result[i], temp_eval_result[i])

            progress.update(task_id=task_id, advance=1)

        if self._judge_main_process():
            for i, result in enumerate(self.all_eval_result):
                merge_list = []
                for rank_list in result:
                    merge_list = merge_list + rank_list
                self.all_eval_result[i] = merge_list

            return calculate_metrics(
                self.logger,
                self.args["metrics"],
                self.all_eval_result,
            )
        else:
            return {"Not Main Process": 0.0}

    # 预测一轮辅助函数
    def _predict_one_epoch(self, task_id, dataloader):
        self.model.eval()

        progress.reset(task_id)

        result_list = []

        for _, eval_data in enumerate(dataloader):
            real_data, eval_result = self._one_batch(eval_data, "predict")
            temp_json = {}
            temp_json["idx"] = real_data["idx"].item()
            temp_json["premise"] = real_data["premise"][0]
            temp_json["hypothesis"] = real_data["hypothesis"][0]
            if eval_result[-1] == [0]:
                temp_json["label"] = 0
            else:
                temp_json["label"] = 2
            # print(temp_json)
            result_list.append(temp_json)
            progress.update(task_id=task_id, advance=1)
        return result_list

    # 处理一小批数据的辅助函数（训练与测试通用）
    def _one_batch(self, data, mode):
        data_need_to_cuda, data_cpu = data
        for data_name in data_need_to_cuda:
            data_need_to_cuda[data_name] = data_need_to_cuda[data_name].to(self.gpu_id)

        if mode == "train":
            self.optimizer.zero_grad()  # zero grad
            loss_single = self.model(mode, self.criterion, **data_need_to_cuda)

            # self.logger.info(self.optimizer.state_dict()['param_groups'][0]['lr'])

            # backward
            loss_single.backward()
            self.optimizer.step()
            self.scheduler.step()
            return loss_single.item()

        elif mode == "eval":
            with torch.no_grad():
                return data_cpu, self.model(mode, self.criterion, **data_need_to_cuda)
        elif mode == "predict":
            with torch.no_grad():
                return data_cpu, self.model(mode, self.criterion, **data_need_to_cuda)

    def _judge_whether_save(self, task_id, epoch, metrics):
        if metrics[self.args["main_metric"]] > self.best_metric["eval"]["metric"]:
            self.logger.info(
                f"Best Metrics Now: {metrics[self.args['main_metric']]} > Best Metrics Before: {self.best_metric['eval']['metric']}"
            )
            self.best_metric["eval"]["epoch"] = epoch
            self.best_metric["eval"]["metric"] = metrics[self.args["main_metric"]]
            progress.update(task_id=task_id, best=self.best_metric["eval"])

            if self.args["model_path"]["save"]:
                self._save_snapshot(epoch)
            self.early_stop_sign.append(0)
        else:
            self.early_stop_sign.append(1)
            # 早停，无法进行进程间的同步
            if sum(self.early_stop_sign) == self.args["early_stop"]:
                self.logger.info(
                    f'The Effect of last {self.args["early_stop"]} epochs has not improved! Early Stop!'
                )
                self.logger.info(f"Best Metric: {self.best_metric}")
                return True
        return False

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
        }
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        torch.save(
            snapshot,
            os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"]),
        )
        self.logger.info(
            f"Epoch {epoch} Training snapshot saved at {self.snapshot_path}"
        )

    def _load_snapshot(self):
        if not os.path.exists(
            os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"])
        ):
            self.logger.warning(
                f"{self.snapshot_path} not exist, Initialize from Scratch"
            )
            return None
        if self.gpu_id != None:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(
                os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"]),
                map_location=loc,
            )
        else:
            snapshot = torch.load(
                os.path.join(self.snapshot_path, self.args["model_path"]["snapshot"])
            )

        self.logger.info(f"Resuming from snapshot at Epoch {snapshot['EPOCHS_RUN']}")
        return snapshot

    def _write_tensorboard(
        self, epoch, loss, metrics_train, metrics_eval, metrics_test
    ):
        self.tensorboard_writer.add_scalar(f"Loss", loss, epoch)
        for i in list(metrics_train.keys()):
            if metrics_test == {}:
                self.tensorboard_writer.add_scalars(
                    f"{i}",
                    {"Train_" + i: metrics_train[i], "Eval_" + i: metrics_eval[i]},
                    epoch + 1,
                )
            else:
                self.tensorboard_writer.add_scalars(
                    f"{i}",
                    {
                        "Train_" + i: metrics_train[i],
                        "Eval_" + i: metrics_eval[i],
                        "Test_" + i: metrics_test[i],
                    },
                    epoch + 1,
                )

    def _judge_main_process(self):
        if self.args["device"] == "CPU" or (
            self.args["device"] != "CPU"
            and self.args["device"] == self.args["global_device"]
        ):
            return True
        return False
