
# 深度学习模型快速配置模板（自用）
> 以文本分类任务为例

## Requirements
```python
pip install -r requirements.txt
```

## Useage

### 1.在 config 目录添加一份配置，修改一些需要自定义的配置

### 2.添加 datasets 下数据集文件，并根据数据集特征，修改 dataset.py 文件
* 修改 InputExample, InputFeature, load_example, convert_examples_to_features

### 3.定义 model

### 4.定义 trainer，一般只关心 损失计算 的部分

### 5.定义 metric 处理评价指标，以及解码输出

### 6.最后在 main 中整体调整一些需要自定义的地方，大部分不用改，如果有新增配置需要相应修改一下。