# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2023
#
# @author: gaohongkui <gaohongkui1021@163.com>
# @date: 2023/4/5
#
import torch
from torch import nn
from transformers import BertModel, PreTrainedModel, PretrainedConfig

from utils import load_tokenizer, load_config


class MyModel(PreTrainedModel):
    def __init__(self, pretrained_config, custom_config):
        super(MyModel, self).__init__(pretrained_config)
        self.custom_config = custom_config

        self.bert = BertModel.from_pretrained(self.custom_config.model.pretrained_model)
        self.tokenizer = load_tokenizer(self.custom_config)

        # 定义分类层
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.custom_config.model.num_classes)

    def forward(self, **kwargs):
        # 获取输入
        input_ids = kwargs.get('input_ids')
        attention_mask = kwargs.get('attention_mask')
        token_type_ids = kwargs.get('token_type_ids')
        labels = kwargs.get('labels', None)

        # 获取 bert 输出
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 取 pooler_output
        pooler_output = bert_output[1]  # [batch_size, hidden_size]
        # dropout
        pooler_output = self.dropout(pooler_output)
        # 分类
        logits = self.classifier(pooler_output)  # [batch_size, num_classes]

        output = {"logits": logits}

        return output


if __name__ == '__main__':
    # 测试模型
    config = load_config("config/baseline.yaml")
    pretrained_config = PretrainedConfig.from_pretrained(config.model.pretrained_model)
    model = MyModel(pretrained_config, config)
    print(model)
    # 模拟输入
    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.randint(0, 2, (2, 10))
    token_type_ids = torch.randint(0, 2, (2, 10))
    labels = torch.randint(0, 2, (2,))
    # 前向传播
    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
    print(output)
