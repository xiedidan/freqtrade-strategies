# WSL2 GPU环境配置指南

本指南将帮助你在Windows 11 + WSL2环境中配置RAPIDS cuDF，以便使用RTX 3060Ti进行GPU加速回测。

## 前置条件检查

### 1. 确认Windows版本

```powershell
# 在PowerShell中运行
winver
```

需要：Windows 11 或 Windows 10 版本 21H2 或更高

### 2. 确认WSL2已安装

```powershell
# 检查WSL版本
wsl --version

# 如果未安装，运行：
wsl --install -d Ubuntu-24.04
```

### 3. 确认NVIDIA驱动已安装

```powershell
# 在PowerShell中运行
nvidia-smi
```

应该能看到你的RTX 3060Ti信息。如果没有，请从NVIDIA官网下载最新驱动：
https://www.nvidia.com/Download/index.aspx

## 快速设置（推荐）

### 步骤1: 进入WSL2

```powershell
# 在PowerShell中运行
wsl
```

### 步骤2: 克隆或同步项目

```bash
# 如果项目在Windows中，可以直接访问
cd /mnt/d/project/freqtrade-strategies

# 或者在WSL中克隆
git clone https://github.com/xiedidan/freqtrade-strategies.git
cd freqtrade-strategies
```

### 步骤3: 运行自动设置脚本

```bash
# 给脚本执行权限
chmod +x scripts/setup_wsl_gpu_env.sh

# 运行设置脚本
./scripts/setup_wsl_gpu_env.sh
```

脚本会自动完成：
- 安装Miniconda
- 创建freqtrade-gpu环境
- 安装RAPIDS cuDF
- 安装Freqtrade
- 安装项目依赖

**注意**: 首次运行如果安装了Miniconda，需要重启终端后再次运行脚本。

### 步骤4: 验证安装

```bash
# 激活环境
conda activate freqtrade-gpu

# 运行验证脚本
python scripts/test_gpu_setup.py
```

如果所有检查都通过，你就可以开始使用GPU加速了！

## 手动设置（高级用户）

如果自动脚本遇到问题，可以手动执行以下步骤：

### 1. 安装Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 重启终端
source ~/.bashrc
```

### 2. 创建Conda环境

```bash
conda create -n freqtrade-gpu python=3.12 -y
conda activate freqtrade-gpu
```

### 3. 安装RAPIDS cuDF

```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.12 \
    python=3.12 \
    cuda-version=12.5 \
    -y
```

### 4. 验证cuDF

```bash
python -c "import cudf; print(f'cuDF version: {cudf.__version__}')"
```

### 5. 安装Freqtrade

```bash
pip install freqtrade==2025.12
```

### 6. 安装项目依赖

```bash
pip install -r requirements-custom.txt
```

## 测试GPU加速

### 1. 测试DataFrame后端

```bash
conda activate freqtrade-gpu
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
python tests/benchmark_dataframe_backend.py
```

这将对比pandas和cuDF的性能，你应该能看到显著的加速效果。

### 3. 测试回测

```bash
# 下载测试数据（如果还没有）
freqtrade download-data \
    --config configs/HourBreakout1.json \
    --timerange 20240101-20240131 \
    --timeframe 1m 5m 1h

# 运行回测
python -m parallel_backtest \
    --config configs/HourBreakout1.json \
    --strategy HourBreakout1 \
    --timerange 20240101-20240131
```

## 性能优化建议

### 1. 监控GPU使用

在另一个终端中运行：
```bash
watch -n 1 nvidia-smi
```

### 2. 调整批处理大小

如果遇到GPU内存不足，可以：
- 减少时间范围
- 减少交易对数量
- 使用更小的数据集

### 3. 混合使用策略

对于小数据集，pandas可能更快。可以在代码中动态选择：

```python
from parallel_backtest.dataframe_backend import DataFrameBackend

# 根据数据量选择后端
if len(dataframe) > 10000:
    DataFrameBackend.initialize('cudf')
else:
    DataFrameBackend.initialize('pandas')
```

## 故障排除

### 问题1: nvidia-smi在WSL中不可用

**症状**:
```bash
nvidia-smi
# Command not found
```

**解决方案**:
1. 确认Windows主机已安装NVIDIA驱动
2. 更新WSL: `wsl --update` (在PowerShell中)
3. 重启WSL: `wsl --shutdown` (在PowerShell中)

### 问题2: cuDF导入失败

**症状**:
```python
ImportError: libcuda.so.1: cannot open shared object file
```

**解决方案**:
```bash
# 添加CUDA库路径
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 问题3: Conda环境解析失败

**症状**:
```
Solving environment: failed
```

**解决方案**:
```bash
# 使用libmamba求解器
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# 重试安装
conda install -c rapidsai cudf=24.12
```

### 问题4: GPU内存不足

**症状**:
```
RuntimeError: out of memory
```

**解决方案**:
1. 减少数据量
2. 关闭其他GPU应用
3. 回退到pandas:
   ```python
   DataFrameBackend.initialize('pandas')
   ```

### 问题5: 性能没有提升

**可能原因**:
1. 数据量太小（<10,000行）
2. CPU-GPU传输开销
3. 操作不适合GPU加速

**解决方案**:
- 使用更大的数据集测试
- 批量处理多个操作
- 查看基准测试结果确定哪些操作受益最大

## 环境管理

### 激活环境

```bash
conda activate freqtrade-gpu
```

### 停用环境

```bash
conda deactivate
```

### 删除环境

```bash
conda env remove -n freqtrade-gpu
```

### 导出环境

```bash
conda env export > environment.yml
```

### 从导出文件创建环境

```bash
conda env create -f environment.yml
```

## 下一步

环境配置完成后，你可以：

1. **运行完整基准测试**
   ```bash
   python tests/benchmark_dataframe_backend.py
   ```

2. **集成到策略中**
   - 修改HourBreakout1策略使用DataFrame抽象层
   - 测试策略在GPU加速下的性能

3. **优化性能**
   - 识别性能瓶颈
   - 针对性优化关键操作

4. **生产部署**
   - 创建Docker镜像
   - 配置自动化回测流程

## 参考资源

- [RAPIDS官方文档](https://docs.rapids.ai/)
- [WSL2 GPU支持](https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [cuDF API文档](https://docs.rapids.ai/api/cudf/stable/)
- [Freqtrade文档](https://www.freqtrade.io/)

## 获取帮助

如果遇到问题：
1. 查看本文档的故障排除章节
2. 运行 `python scripts/test_gpu_setup.py` 诊断问题
3. 查看RAPIDS社区: https://rapids.ai/community.html
4. 在项目仓库提交Issue

祝你GPU加速回测顺利！🚀
