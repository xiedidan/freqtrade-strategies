# GPU加速指南

本指南介绍如何使用RAPIDS cuDF对Freqtrade策略进行GPU加速。

## 概述

通过使用RAPIDS cuDF替换pandas，可以利用GPU加速数据处理操作，显著提升回测和实时交易的性能。

**预期性能提升**:
- 数据处理速度：3-10倍
- 回测总时间：2-5倍（取决于策略复杂度）

## 系统要求

### 硬件要求
- **GPU**: NVIDIA Volta或更高（计算能力7.0+）
- **GPU内存**: 建议8GB以上
- **系统内存**: 建议16GB以上

### 软件要求
- **操作系统**: 
  - Linux (Ubuntu 20.04+, Rocky Linux 8+)
  - Windows 11 with WSL2
- **CUDA**: 11.2, 11.4, 11.5, 11.8, 12.0, 12.2, 或 12.5
- **Python**: 3.9-3.12
- **NVIDIA驱动**: 支持所需CUDA版本的最新驱动

## 安装指南

### 方案1: WSL2 (Windows用户推荐)

#### 1. 安装WSL2和Ubuntu

```powershell
# 在PowerShell中以管理员身份运行
wsl --install -d Ubuntu-24.04
```

#### 2. 安装NVIDIA驱动

在Windows主机上安装最新的NVIDIA驱动（不是在WSL2中）：
- 下载地址: https://www.nvidia.com/Download/index.aspx

#### 3. 在WSL2中安装CUDA Toolkit

```bash
# 进入WSL2
wsl

# 安装CUDA Toolkit (不包含驱动)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-5
```

#### 4. 安装Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 重启终端或运行: source ~/.bashrc
```

#### 5. 创建RAPIDS环境

```bash
# 创建新的conda环境
conda create -n freqtrade-gpu python=3.12

# 激活环境
conda activate freqtrade-gpu

# 安装RAPIDS cuDF
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 python=3.12 cuda-version=12.5

# 安装Freqtrade
pip install freqtrade==2025.12

# 安装项目依赖
pip install -r requirements-custom.txt
```

### 方案2: Linux原生安装

#### 1. 安装NVIDIA驱动和CUDA

```bash
# Ubuntu示例
sudo apt-get update
sudo apt-get install -y nvidia-driver-535
sudo apt-get install -y cuda-toolkit-12-5
```

#### 2. 安装RAPIDS

按照WSL2方案的步骤4-5进行。

### 方案3: Docker (推荐用于生产环境)

```bash
# 拉取RAPIDS镜像
docker pull rapidsai/rapidsai:24.12-cuda12.5-runtime-ubuntu22.04-py3.12

# 运行容器
docker run --gpus all -it \
    -v $(pwd):/workspace \
    rapidsai/rapidsai:24.12-cuda12.5-runtime-ubuntu22.04-py3.12

# 在容器中安装Freqtrade
pip install freqtrade==2025.12
```

## 使用方法

### 1. 验证安装

```python
# 测试cuDF是否可用
python -c "import cudf; print(f'cuDF version: {cudf.__version__}')"

# 运行后端信息检查
python -m parallel_backtest.dataframe_backend
```

预期输出：
```
==================================================
DataFrame Backend Information
==================================================
Active Backend: cudf
cuDF Available: True
Using cuDF: True
Pandas Version: 2.3.3
cuDF Version: 24.12.0
==================================================
```

### 2. 运行性能基准测试

```bash
# 对比pandas和cuDF性能
python tests/benchmark_dataframe_backend.py
```

这将输出详细的性能对比数据。

### 3. 在策略中使用

DataFrame抽象层会自动检测cuDF是否可用：

```python
from parallel_backtest.dataframe_backend import DataFrameBackend

# 自动使用最佳后端（cuDF如果可用，否则pandas）
DataFrameBackend.initialize('auto')

# 或强制使用特定后端
DataFrameBackend.initialize('cudf')  # 强制使用cuDF
DataFrameBackend.initialize('pandas')  # 强制使用pandas
```

### 4. 运行回测

```bash
# 使用GPU加速的回测
python -m parallel_backtest \
    --config configs/HourBreakout1.json \
    --strategy HourBreakout1 \
    --timerange 20240101-20241231
```

## 性能基准

### 测试环境
- CPU: AMD Ryzen 9 / Intel i9
- GPU: NVIDIA RTX 3080 / RTX 4090
- 数据量: 100,000行 (约69天的1分钟数据)

### 基准测试结果 (Windows + pandas)

| 操作 | pandas (ms) | 说明 |
|------|-------------|------|
| DataFrame创建 | 0.73 | 创建OHLCV数据框 |
| 滚动均值(MA5) | 1.95 | 计算5周期移动平均 |
| Shift操作 | 0.12 | 数据偏移 |
| 条件筛选 | 1.15 | 基于条件过滤数据 |
| 列运算 | 0.79 | 算术运算 |
| DataFrame合并 | 2.35 | 多时间框架数据合并 |

### 预期cuDF性能提升

基于RAPIDS官方基准测试，预期性能提升：

| 操作 | 预期加速比 |
|------|-----------|
| DataFrame创建 | 2-3x |
| 滚动均值 | 5-10x |
| Shift操作 | 3-5x |
| 条件筛选 | 4-8x |
| 列运算 | 3-6x |
| DataFrame合并 | 3-7x |

**注意**: 实际性能提升取决于GPU型号、数据量和操作复杂度。

## 故障排除

### cuDF导入失败

```python
ImportError: No module named 'cudf'
```

**解决方案**:
1. 确认在正确的conda环境中
2. 重新安装cuDF: `conda install -c rapidsai cudf`

### CUDA版本不匹配

```
RuntimeError: CUDA version mismatch
```

**解决方案**:
1. 检查CUDA版本: `nvcc --version`
2. 安装匹配的cuDF版本
3. 参考: https://docs.rapids.ai/install/

### GPU内存不足

```
RuntimeError: out of memory
```

**解决方案**:
1. 减少批处理大小
2. 使用更小的时间范围
3. 回退到pandas: `DataFrameBackend.initialize('pandas')`

### WSL2 GPU不可用

```bash
# 检查GPU是否可见
nvidia-smi
```

如果失败：
1. 确认Windows主机已安装最新NVIDIA驱动
2. 更新WSL2: `wsl --update`
3. 重启WSL2: `wsl --shutdown`

## 最佳实践

### 1. 混合使用策略

对于小数据集（<10,000行），pandas可能更快（避免GPU传输开销）。建议：

```python
# 根据数据量自动选择后端
if len(dataframe) > 10000:
    DataFrameBackend.initialize('cudf')
else:
    DataFrameBackend.initialize('pandas')
```

### 2. 批量处理

将多个操作组合在一起，减少CPU-GPU数据传输：

```python
# 好的做法：批量操作
df['ma5'] = df['close'].rolling(5).mean()
df['ma10'] = df['close'].rolling(10).mean()
df['ma20'] = df['close'].rolling(20).mean()

# 避免：频繁的小操作
```

### 3. 内存管理

及时释放不需要的DataFrame：

```python
# 转换回pandas后释放GPU内存
result_df = DataFrameBackend.to_pandas(gpu_df)
del gpu_df  # 释放GPU内存
```

## 限制和注意事项

1. **Windows原生不支持**: 必须使用WSL2
2. **单GPU支持**: 当前实现仅支持单GPU
3. **API兼容性**: cuDF不是100%兼容pandas，某些高级功能可能不可用
4. **内存限制**: GPU内存通常小于系统内存，需要注意数据量
5. **传输开销**: 小数据集可能因CPU-GPU传输开销而变慢

## 相关资源

- [RAPIDS官方文档](https://docs.rapids.ai/)
- [cuDF API参考](https://docs.rapids.ai/api/cudf/stable/)
- [RAPIDS安装指南](https://docs.rapids.ai/install/)
- [WSL2 GPU支持文档](https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [NVIDIA CUDA文档](https://docs.nvidia.com/cuda/)

## 反馈和支持

如遇到问题或有改进建议，请：
1. 查看故障排除章节
2. 在项目仓库提交Issue
3. 参考RAPIDS社区支持: https://rapids.ai/community.html
