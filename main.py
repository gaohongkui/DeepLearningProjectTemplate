# !/usr/bin/python
# -*- coding: utf-8 -*-
#
# Copyright @2023
#
# @author: gaohongkui <gaohongkui1021@163.com>
# @date: 2023/4/4
#
import os

import jsonlines
import torch
from loguru import logger
from transformers import TrainingArguments, default_data_collator, PretrainedConfig, EarlyStoppingCallback

from dataset import MyDataset
from metric import MyMetric
from model import MyModel
from trainer import MyTrainer
from utils import set_seed, set_gpus, load_config


# 统一训练、测试、预测的入口
def run(config):
    set_seed(config.runner.seed)
    set_gpus(config.gpu_list)

    # 加载训练参数，注意这里的参数是从配置文件中读取的
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        do_train=config.is_train,
        do_eval=config.is_eval,
        do_predict=config.is_predict,

        no_cuda=config.device == 'cpu',
        metric_for_best_model=config.runner.metric_for_best_model,
        greater_is_better=config.runner.greater_is_better,
        gradient_accumulation_steps=config.runner.gradient_accumulation_steps,

        num_train_epochs=config.runner.num_train_epochs,
        per_device_train_batch_size=config.data.per_device_train_batch_size,
        per_device_eval_batch_size=config.data.per_device_eval_batch_size,
        dataloader_num_workers=config.data.dataloader_num_workers,
        label_names=list(config.data.label_names),

        # optimizer
        optim=config.optimizer.optim,
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=config.optimizer.weight_decay,
        # scheduler
        lr_scheduler_type=config.scheduler.lr_scheduler_type,
        warmup_ratio=config.scheduler.warmup_ratio,
        warmup_steps=config.scheduler.warmup_steps,

        # checkpoint
        save_strategy=config.checkpoint.save_strategy,
        save_steps=config.checkpoint.save_steps,
        save_total_limit=config.checkpoint.save_total_limit if config.checkpoint.save_total_limit > 0 else None,
        evaluation_strategy=config.checkpoint.evaluation_strategy,
        eval_steps=config.checkpoint.eval_steps,

        # logging
        report_to=list(config.runner.report_to),
        # logging_dir=config.runner.logging_dir if config.runner.logging_dir is not None else os.path.join(config.output_dir, "runs"),

        load_best_model_at_end=True
    )
    # 加载 callbacks
    callbacks = []
    if config.runner.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.runner.early_stopping_patience,
                                               early_stopping_threshold=config.runner.early_stopping_threshold))

    # 加载数据集
    train_dataset = MyDataset(config, 'train')
    eval_dataset = MyDataset(config, 'eval')
    # 加载模型
    pretrained_config = PretrainedConfig.from_pretrained(config.model.pretrained_model)  # 预训练模型的配置
    pretrained_config.update(config.model)  # 更新配置
    model = MyModel(pretrained_config)
    # 加载 metric
    metric = MyMetric(config)
    # 加载 trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric.compute_metrics,

        callbacks=callbacks
    )

    best_model_path = os.path.join(config.output_dir, "best")
    # 训练
    if config.is_train:
        logger.info('*** Training ***')
        train_results = trainer.train(resume_from_checkpoint=config.checkpoint.resume_from_checkpoint)
        logger.info(f"*** Train results: ***".center(20))
        logger.info(train_results)
        trainer.save_metrics("train", train_results.metrics)
        # 最优模型保存
        # save_best_model(config, model)
        trainer.save_model(best_model_path)

    with torch.no_grad():
        # 加载最优模型
        model = MyModel.from_pretrained(best_model_path, config=pretrained_config)
        trainer = MyTrainer(
            model=model,
            args=training_args,
            data_collator=default_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric.compute_metrics,

            callbacks=callbacks
        )
        # 验证
        if config.is_eval:
            logger.info('*** Evaluating ***')
            eval_results = trainer.evaluate(eval_dataset)
            logger.info(f"*** Eval results: ***".center(20))
            logger.info(eval_results)
            trainer.save_metrics("eval", eval_results)
        # 测试
        if config.is_test:
            logger.info('*** Testing ***')
            test_datasets = MyDataset(config, 'test')
            test_results = trainer.evaluate(test_datasets)
            logger.info(f"*** Test results: ***".center(20))
            logger.info(test_results)
            trainer.save_metrics("test", test_results)
        # 预测
        if config.is_predict:
            logger.info('*** Predicting ***')
            predict_datasets = MyDataset(config, 'predict')
            predict_results = trainer.predict(predict_datasets)
            # 解码并保存预测结果
            predict_results = metric.decode(predict_results, predict_datasets)
            save_path = os.path.join(config.output_dir, "best/pred_results.jsonl")
            with jsonlines.open(save_path, "w") as fout:
                fout.write_all(predict_results)
            logger.info(f"*** pred results saved to {save_path} ***".center(20))


if __name__ == '__main__':
    config = load_config("config/baseline.yaml")
    run(config)
