# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2023
#
# @author: gaohongkui <gaohongkui1021@163.com>
# @date: 2023/4/3
#
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer

root_dir = os.path.abspath(os.path.dirname(__file__))


def load_config(file_path):
    default_path = Path(os.path.join(root_dir, "config/defaults.yaml"))
    default_config = OmegaConf.create(yaml.load(default_path.open(), Loader=yaml.FullLoader),
                                      flags={"allow_objects": True})

    file_path = Path(os.path.join(root_dir, file_path))  # 如果传入的是相对路径，Path 取相对于工作空间的目录
    config = OmegaConf.create(yaml.load(file_path.open(), Loader=yaml.FullLoader), flags={"allow_objects": True})
    logger.info(f"load config from {file_path} success")
    config = OmegaConf.merge(default_config, config)

    # 整理 path 和 注入变量
    base_path = config.base_path
    for path in config.data_path:
        config.data_path[path] = os.path.join(base_path, config.data_path[path]) if config.data_path[path] else None

    output_dir = os.path.join(base_path, "outputs", config.model_version)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config.output_dir = output_dir

    logger.info(f"merge config from baseline and {file_path} success")

    logger.debug("merge model config detail:\n" + OmegaConf.to_yaml(config))
    return config


def set_seed(seed):
    """
    Sets random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def set_gpus(gpus):
    if not gpus:
        return
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def load_tokenizer(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model)
    if config.model.max_seq_length:
        tokenizer.model_max_length = config.model.max_seq_length

    return tokenizer


def save_best_model(config, model, save_path=None):
    if save_path is None:
        if not os.path.exists(config.output_dir + "/best"):
            os.mkdir(config.output_dir + "/best")
        save_path = config.output_dir + "/best/model_state_dict.pt"
    torch.save(model.state_dict(), save_path)


def load_best_model(config, model, save_path=None, device=None):
    if save_path is None:
        save_path = config.output_dir + "/best/model_state_dict.pt"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(save_path, map_location=device))


if __name__ == '__main__':
    baseline = load_config("config/baseline.yaml")
    print(type(baseline.runner.report_to))
    print(list(baseline.runner.report_to))
    print(isinstance(baseline.runner.report_to, list))
