import os
import yaml
import torch
from loguru import logger
from torch.distributed import init_process_group
from .set_logger import DDPLogger
from .set_seed import set_seed

ENVIRONMENT_VARIABLES = [
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
]


def get_config(args):
    # 获取当前配置的设备信息
    device, global_device = get_device(args)

    # 设置日志
    logger_myself = DDPLogger(args.datetime, global_device, device).logger

    logger_datetime = args.datetime

    # 查看环境变量
    if device != "CPU":
        ddp_variables(logger_myself)

    # 读取配置文件
    with open(args.config_file, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # 设置随机种子
    set_seed(args["seed"])

    args["logger"] = logger_myself
    args["device"] = device
    args["global_device"] = global_device
    args["world_size"] = int(os.environ["WORLD_SIZE"])
    args["datetime"] = logger_datetime

    return args


def get_device(args):
    gpu_list = [int(i) for i in args.gpu.split(",")]
    logger.info(f"GPU: {gpu_list}")
    if gpu_list[0] != -1:
        ddp_setup()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = "GPU-{}".format(gpu_list[int(os.environ["LOCAL_RANK"])])
        global_device = "GPU-{}".format(gpu_list[0])

    else:
        device = "CPU"
        global_device = "CPU"
    return device, global_device


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def ddp_variables(logger):
    for name in ENVIRONMENT_VARIABLES:
        logger.info(f"{name}:{os.environ[name]}")
