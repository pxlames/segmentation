# simple-unet-2d
simple unet with NeurIPS'19 topoloss

## Python 环境

使用项目根目录下的 `environment.yml` 创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate environment   # 若 yml 未指定 name，默认环境名为 environment
```

### 主要依赖

| 类型 | 包名 | 版本 |
|------|------|------|
| 基础 | Python | 3.8.19 |
| | numpy | 1.24.3 |
| | pandas | 2.0.2 |
| | scipy | 1.10.1 |
| 深度学习 | PyTorch | 2.3.0 |
| | torchvision | 0.16.1 / 0.18.0 (pip) |
| | cudatoolkit | 11.8.0 |
| | torchaudio | 2.3.0 |
| 图像/可视化 | opencv | 4.6.0 / 4.7.0.72 (pip) |
| | matplotlib | 3.7.2 |
| 可选 | keras | 2.13.1 |
| | dgl | 2.1.0（图神经网络） |
| | torch-geometric | 2.6.1 |
| | pytorch-lightning | 1.5.0 |
| | wandb | 0.13.5 |
| | tqdm | 4.65.2 |

**说明：** 子模块 `vessel_salience`、`fundus-vessels-toolkit`、`pyvane` 可能有各自的依赖，首次使用前需单独安装（如 `pip install -e vessel_salience` 等）。

---

**Commands:**

* Make sure to populate `train.json` and `test.json` with appropriate hyprerparameters

**Train:**
```
CUDA_VISIBLE_DEVICES=3 python3 main.py --params ./datalists/DRIVE/train.json
```
* Ensure `crop_size` in `train.json` is divisible by 16

**Test/Inference:**
```
CUDA_VISIBLE_DEVICES=4 python3 main.py --params ./datalists/DRIVE/test.json
```
**Compute Evaluation Metrics (Quantitative Results):**
```
python3 compute-eval-metrics.py
```
**Dataset properties:**

GT: Foreground should be 255 ; Background should be 0

* First do pretrain (1000-2000 epochs) by setting `"topo_weight": 0` in `train.json`
* Then, rainload the best model from pretrain and t using topoloss by setting `topo_weight` to a non-zero value. Change the `output_folder` and `checkpoint_restore` in `train.json` too
