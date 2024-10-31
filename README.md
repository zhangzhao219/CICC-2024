# 我是大明星队 赛题1 代码运行说明

# 运行环境

GPU： A800 80G * 8 CUDA 12.4

Python 3.11.10

依赖包见 requirements.txt，主要功能的包需完整且功能一致

# 复现方式

## 使用开源大模型对数据进行增强

### 下载大模型权重

- [Yi-1.5-34B-Chat-16K](https://www.modelscope.cn/models/01ai/Yi-1.5-34B-Chat-16K)
- [Qwen2.5-72B-Instruct]([https://www.modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct)
- [Qwen2-72B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-72B-Instruct)

下载好后更改```infer_zero_shot_reason.py```里面的```MODEL_PATH```变量为模型的路径

在```cicc```目录下运行```bash run.sh```，运行结束后会在```data```目录下生成数据文件：

```bash
test_data_Qwen2___5-72B-Instruct_analysis.json
test_data_Qwen2-72B-Instruct_analysis.json
test_data_Yi-1___5-34B-Chat-16K_analysis.json
train_data_Qwen2___5-72B-Instruct_analysis.json
train_data_Qwen2-72B-Instruct_analysis.json
train_data_Yi-1___5-34B-Chat-16K_analysis.json
val_data_Qwen2___5-72B-Instruct_analysis.json
val_data_Qwen2-72B-Instruct_analysis.json
val_data_Yi-1___5-34B-Chat-16K_analysis.json
```

## 使用增强的数据训练下游小模型

下载预训练ERNIE权重：[ernie-3.0-xbase-zh](https://www.modelscope.cn/models/tiansz/ernie-3.0-xbase-zh)


### 训练基于ERNIE预训练模型的下游模型

更改```config/CICC_ernie.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```
在```cicc/Pytorch-template-main```目录下运行```bash runsh/run_template_ernie.sh```进行模型训练

### 训练基于TextCNN模型的下游模型
更改```config/CICC_textcnn.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```

在```cicc/Pytorch-template-main```目录下运行```bash runsh/run_template_textcnn.sh```进行模型训练

### 模型权重保存位置

运行结束后生成的模型权重文件在```cicc/Pytorch-template-main/save_models```的运行时间戳的文件夹中

（目前训练好的权重文件已经保存在该文件夹中）

## 推理下游小模型得到结果

更改```config/CICC_ernie_qwen25_test.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```

更改```config/CICC_ernie_qwen2_test.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```

更改```config/CICC_ernie_yi_test.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```

更改```config/CICC_textcnn_qwen2_test.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```

更改```config/CICC_textcnn_qwen25_test.yaml```内部文件夹的路径：
```yaml
data_path:
  data_dir: "{cicc文件夹的绝对路径}"
model_path:
  pretrained_model_dir: "{ERNIE权重的绝对路径（不包括ERNIE本身文件夹）}"
```

在```cicc/Pytorch-template-main```目录下运行```bash runsh/eval.sh```进行模型推理

### 结果文件生成位置

会在```result```文件夹下生成csv文件：

```bash
val_data_*.csv
test_data_*.csv
```

（目前生成好的csv文件已经保存在该文件夹中）

## 集成多个模型的输出结果

在```cicc```目录下运行```python merge.py```进行模型集成，会在```cicc```目录下生成```test_label.json```文件，即为最终结果

（目前生成好的```test_label.json```文件已经保存在此位置）
