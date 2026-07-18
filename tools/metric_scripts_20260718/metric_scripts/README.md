# 当前使用的 3MTI 指标脚本

## 文件

- `scripts/eval_image_folder_3mti_style.py`：读取预测图与 HR 真值目录，逐图计算并汇总指标。
- `rg_rmf/metrics.py`：PSNR、SSIM 及其余 3MTI 指标的核心实现。
- `rg_rmf/__init__.py`：Python 包入口。

以上三个文件均为 2026-07-16 从 4090 服务器原样复制的版本。

## 只计算 PSNR / SSIM

在本目录执行：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/eval_image_folder_3mti_style.py \
  --pred-dir /path/to/pred \
  --ref-dir /path/to/hr_thermal \
  --output-dir /path/to/metrics_psnr_ssim \
  --metrics train
```

预测图和真值图按照相同文件名配对。输出包括：

- `metrics_per_image.csv`
- `summary.json`

## 当前论文表格口径

PSNR 和 SSIM 使用 `pyiqa==0.1.7`：

```python
pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr")
pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr")
```

- 在 Y 通道上计算；
- 默认不裁边；
- 默认不改变成对图像尺寸；
- 最终结果为全部图像逐图指标的算术平均。

4090 当前环境版本：

```text
pyiqa       0.1.7
torch       2.0.1
torchvision 0.15.2
numpy       1.26.0
```

## 与训练日志中的验证值区别

FiDeSR 训练期间 `train_fidesr.py` 内置的轻量验证使用 RGB 张量上的自定义 PSNR/SSIM，并在计算前执行配置中的 AdaIN 对齐；它用于挑选 `best_psnr.pkl`。

论文表格和全指标测试使用本包中的 3MTI-style、Y 通道 `pyiqa` 口径，因此训练日志中的验证值与最终表格数值不要求完全相同。

## 服务器原路径

```text
/data1/matengyu/cch/RG_RMF/scripts/eval_image_folder_3mti_style.py
/data1/matengyu/cch/RG_RMF/rg_rmf/metrics.py
```
