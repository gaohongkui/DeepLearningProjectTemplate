# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2023
#
# @author: gaohongkui <gaohongkui1021@163.com>
# @date: 2023/4/4
#
# 加载数据集与特征转换等
import json
from typing import List

import jsonlines
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, default_data_collator

from utils import load_tokenizer, load_config


class MyDataset(Dataset):
    def __init__(self, config: DictConfig, mode: str):
        self.config = config
        data_path = {
            "train": self.config.data_path.train_file,
            "eval": self.config.data_path.eval_file,
            "test": self.config.data_path.test_file,
            "predict": self.config.data_path.predict_file
        }
        self.tokenizer = load_tokenizer(self.config)
        self.load_labels(self.config.data_path.labels_file)

        self.predict_mode = True if mode == "predict" else False
        self.examples = load_examples(data_path[mode], self.predict_mode)
        self.data_list = convert_examples_to_features(self.examples, self.tokenizer, self.lable2id, self.predict_mode)
        assert len(self.examples) == len(self.data_list), "确保相等便于后续预测解码，否则需要定义如何找到 feature 对应的 example"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def load_labels(self, labels_file):
        labels = []
        with jsonlines.open(labels_file) as fin:
            for line in fin:
                labels.append(line["label_desc"])
        self.lable2id = {k: v for v, k in enumerate(labels)}
        self.id2label = {k: v for k, v in enumerate(labels)}


class InputExample:
    def __init__(self, text, label_desc=None, item_id=None):
        self.text = text
        self.label_desc = label_desc
        self.item_id = item_id

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True, ensure_ascii=False)


class InputFeature:
    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True, ensure_ascii=False)


def load_examples(input_file: str, predict_mode: bool):
    # 加载数据集文件
    examples = []
    with jsonlines.open(input_file) as fin:
        for idx, item in enumerate(fin):
            text = item["sentence"]
            label_desc = item["label_desc"] if not predict_mode else None
            item_id = idx
            examples.append(InputExample(text=text,
                                         label_desc=label_desc,
                                         item_id=item_id))
    logger.info(f"load {len(examples)} example from {input_file}")
    return examples


def convert_examples_to_features(examples: List[InputExample], tokenizer: PreTrainedTokenizer, label2id: dict,
                                 predict_mode: bool):
    # 将 examples 转换为输入模型所用的 feature
    features = []
    for idx, example in enumerate(tqdm(examples, "convert examples to features")):
        text = example.text
        inputs = tokenizer(text, truncation=True, padding="max_length")
        tokens = inputs.tokens()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        label = None
        if not predict_mode:
            label = label2id[example.label_desc]

        if idx < 5:
            logger.info(f"*** InputFeature ***")
            logger.info(f"item_id: {example.item_id}")
            logger.info(f"tokens: {tokens}")
            logger.info(f"input_ids: {input_ids}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"token_type_ids: {token_type_ids}")
            logger.info(f"label: {label}")

        features.append(InputFeature(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     label=label))
    logger.info(f"load {len(features)} features")
    return features


if __name__ == '__main__':
    config = load_config("config/baseline.yaml")
    datasets = MyDataset(config, "eval")
    dataloader = DataLoader(dataset=datasets, batch_size=5, collate_fn=default_data_collator)
    for data in dataloader:
        print(data)
        break
