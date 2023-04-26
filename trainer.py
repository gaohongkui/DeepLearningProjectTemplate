# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2023
#
# @author: gaohongkui <gaohongkui1021@163.com>
# @date: 2023/4/6
#
# 继承 Transformers 的 Trainer 类，实现自定义的训练逻辑

from torch import nn
from transformers import Trainer


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     # 自定义训练逻辑
    #     # model 是一个 nn.Module 对象，包含了模型的结构和参数
    #     # inputs 是一个字典，包含了训练数据的多个特征
    #     # 例如，inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]
        # 自定义损失函数
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.num_classes), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
