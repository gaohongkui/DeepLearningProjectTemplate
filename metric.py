# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2023
#
# @author: gaohongkui <gaohongkui1021@163.com>
# @date: 2023/4/6
#
import evaluate
import numpy as np
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput

from dataset import MyDataset


# 使用于 Transformers Trainer 的 Metric 类，计算 F1、Precision、Recall
class MyMetric:
    def __init__(self, config):
        self.config = config

    def get_p_r_f(self, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        return np.array([p, r, f1])

    def compute_metrics(self, outputs: EvalPrediction):
        predictions, labels = outputs.predictions, outputs.label_ids
        logits = predictions[0] if isinstance(predictions, tuple) else predictions  # batch_size * num_labels
        preds = logits.argmax(axis=-1)  # batch_size
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        results = {}
        results.update(precision_metric.compute(predictions=preds, references=labels, average="macro"))
        results.update(recall_metric.compute(predictions=preds, references=labels, average="macro"))
        results.update(f1_metric.compute(predictions=preds, references=labels, average="macro"))
        return results

    def decode(self, output: PredictionOutput, pred_datasets: MyDataset):
        # 解码预测结果
        logits = output.predictions
        preds = logits.argmax(axis=-1)

        id2label = pred_datasets.id2label
        results = []
        for idx, pred in enumerate(preds):
            results.append({"id": idx, "sentence": pred_datasets.examples[idx].text, "label": id2label[pred]})
        return results
