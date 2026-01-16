# TASK-001: CuDF加速HourBreakout1策略实验

## 任务概述

**目标**: 使用RAPIDS cuDF替换pandas进行数据处理，利用GPU加速  
**范围**: HourBreakout1策略的指标计算和数据处理部分  
**预期收益**: 数据处理速度提升5-10倍  
**技术栈**: RAPIDS cuDF, CUDA  
**状态**: 进行中

## 环境分析

### 当前环境
- 操作系统: Windows 11
- Python版本: 3.12.9 (venv)
- Pandas版本: 2.3.3
- Freqtrade版本: 2025.12

### RAPIDS cuDF要求
- GPU: NVIDIA Volta或更高，计算能力7.0+
- CUDA: 11.2, 11.4, 11.5, 11.8, 12.0, 12.2, 12.5
- Windows支持: 需要通过WSL2安装

## 实施方案

### 方案A: WSL2 + cuDF (推荐用于生产)

**优点**:
- 完整的RAPIDS生态系统支持
- 最佳性能
- 官方支持

**缺点**:
- 需要WSL2环境配置
- 环境迁移成本
- 调试复杂度增加

**步骤**:
1. 检查WSL2安装状态
2. 在WSL2中安装CUDA Toolkit
3. 安装RAPIDS cuDF
4. 迁移开发环境到WSL2
5. 修改策略代码支持cuDF

### 方案B: cuDF兼容层 (实验性)

**优点**:
- 保持Windows原生环境
- 代码改动最小
- 快速验证可行性

**缺点**:
- 性能提升有限
- 不是真正的GPU加速
- 仅用于API兼容性测试

**实现**:
创建cuDF兼容包装器，在Windows环境下使用pandas后端，在Linux/WSL2环境下自动切换到cuDF。

### 方案C: 混合方案 (推荐用于实验)

**策略**:
1. 先在Windows环境下创建cuDF兼容接口
2. 编写性能基准测试
3. 在WSL2或Linux环境中进行真实GPU加速测试
4. 对比性能数据

## 实施计划

### Phase 1: 接口抽象 (当前阶段)
- [ ] 创建DataFrame抽象层
- [ ] 识别HourBreakout1中的pandas操作
- [ ] 设计cuDF兼容接口
- [ ] 编写单元测试

### Phase 2: 兼容实现
- [ ] 实现pandas后端
- [ ] 实现cuDF后端（条件导入）
- [ ] 添加自动检测和切换逻辑
- [ ] 集成到策略代码

### Phase 3: 性能测试
- [ ] 创建性能基准测试脚本
- [ ] 在Windows环境测试（pandas后端）
- [ ] 在WSL2/Linux环境测试（cuDF后端）
- [ ] 收集和分析性能数据

### Phase 4: 优化和文档
- [ ] 根据测试结果优化代码
- [ ] 编写使用文档
- [ ] 更新requirements
- [ ] 提交代码审查

## 技术细节

### 需要替换的pandas操作

1. **DataFrame创建和合并**
   ```python
   # pandas
   df = pd.DataFrame(data)
   merged = pd.merge(df1, df2)
   
   # cuDF
   df = cudf.DataFrame(data)
   merged = cudf.merge(df1, df2)
   ```

2. **技术指标计算**
   ```python
   # pandas
   df['ma5'] = df['close'].rolling(5).mean()
   
   # cuDF
   df['ma5'] = df['close'].rolling(5).mean()  # API兼容
   ```

3. **条件筛选**
   ```python
   # pandas
   df[df['close'] > df['ma5']]
   
   # cuDF
   df[df['close'] > df['ma5']]  # API兼容
   ```

### 抽象层设计

```python
# dataframe_backend.py
class DataFrameBackend:
    """Abstract DataFrame backend for pandas/cuDF compatibility"""
    
    @staticmethod
    def create_dataframe(data):
        """Create DataFrame using available backend"""
        pass
    
    @staticmethod
    def merge(df1, df2, **kwargs):
        """Merge DataFrames"""
        pass
    
    @staticmethod
    def to_pandas(df):
        """Convert to pandas DataFrame"""
        pass
```

## 风险和挑战

1. **环境兼容性**: WSL2配置可能遇到问题
2. **API差异**: cuDF不是100%兼容pandas
3. **内存管理**: GPU内存限制
4. **调试难度**: GPU相关错误较难定位

## 成功标准

1. 代码能在pandas和cuDF后端间无缝切换
2. 在GPU环境下性能提升至少3倍
3. 所有现有测试通过
4. 文档完整，易于部署

## 参考资料

- [RAPIDS cuDF文档](https://docs.rapids.ai/api/cudf/stable/)
- [cuDF API参考](https://docs.rapids.ai/api/cudf/stable/api_docs/index.html)
- [RAPIDS安装指南](https://docs.rapids.ai/install/)
- [WSL2 GPU支持](https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

## 进度日志

### 2025-01-16
- ✅ 任务认领并移至进行中
- ✅ 完成环境分析
- ✅ 确定实施方案：混合方案（方案C）
- ✅ 创建任务文档
- ✅ **Phase 1完成**: 接口抽象
  - 创建DataFrame抽象层 (`parallel_backtest/dataframe_backend.py`)
  - 实现pandas/cuDF自动检测和切换
  - 支持常用DataFrame操作（创建、合并、转换等）
- ✅ **Phase 3部分完成**: 性能测试
  - 创建性能基准测试脚本 (`tests/benchmark_dataframe_backend.py`)
  - 完成pandas基准测试（Windows环境）
  - 获得基准性能数据
- ✅ 创建GPU加速使用指南 (`docs/GPU-Acceleration-Guide.md`)
- ✅ 更新requirements-custom.txt，添加cuDF安装说明
- 📝 下一步：将抽象层集成到HourBreakout1策略（Phase 2）

### 当前状态
- **完成度**: 60%
- **阻塞问题**: 无
- **待办事项**:
  1. 集成DataFrame抽象层到HourBreakout1策略
  2. 在WSL2/Linux环境进行真实GPU测试
  3. 收集完整的性能对比数据
  4. 编写集成文档
