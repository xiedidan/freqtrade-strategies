---
inclusion: always
---

# Development Rules / 开发规则

## Language and Documentation Rules / 语言和文档规则

1. **Think in English, code in English** / 用英文思考，用英文编程
   - 所有代码注释和配置文件注释使用英文
   - 变量名、函数名、类名使用英文
   - 提交信息和技术文档使用英文

2. **Respond in Chinese** / 用中文回答
   - 所有面向用户的回复使用中文
   - 所有面向用户的文档使用中文
   - 错误信息和日志可以保留英文以保持技术清晰度

## Environment Management Rules / 环境管理规则

1. **Use venv for Python environment management** / 使用venv进行Python环境管理
   - 始终使用Python虚拟环境（venv）进行开发
   - 创建虚拟环境：`python -m venv venv`
   - 激活虚拟环境：
     - Windows: `venv\Scripts\activate`
     - Linux/Mac: `source venv/bin/activate`
   - 在虚拟环境中安装依赖
   - 保持requirements.txt更新所有依赖

2. **Custom Dependencies Management** / 自定义依赖管理
   - 使用 `requirements-custom.txt` 管理项目特定依赖
   - 该文件包含以下依赖：
     - 并行回测工具（tqdm用于进度条）
     - 测试框架（hypothesis用于属性测试，pytest）
   - 安装自定义依赖：`pip install -r requirements-custom.txt`
   - 将此文件与Freqtrade的主requirements.txt分开
   - 添加新的自定义工具或测试依赖时更新此文件

## Code Quality Standards / 代码质量标准

- 遵循PEP 8 Python代码风格指南
- 使用有意义的英文变量名和函数名
- 为复杂逻辑添加全面的英文注释
- 保持代码结构清晰可读

## WSL2 Environment Rules / WSL2环境规则

1. **WSL2 Access and Usage** / WSL2访问和使用
   - 可以通过 `wsl -e bash -c "command"` 直接在WSL2中执行命令
   - WSL2用户名：xd
   - WSL2密码：xdunix12
   - 项目路径：`/mnt/d/project/freqtrade-strategies`

2. **Sudo Command Execution** / Sudo命令执行
   - 使用 `echo 'xdunix12' | sudo -S command` 方式传递密码
   - 避免交互式密码输入导致命令卡住
   - 示例：`wsl -e bash -c "echo 'xdunix12' | sudo -S apt-get update"`

3. **Conda Environment Management** / Conda环境管理
   - Miniconda安装路径：`~/miniconda3`
   - GPU环境名称：`freqtrade-gpu`
   - 激活环境：`source ~/miniconda3/bin/activate freqtrade-gpu`
   - 完整命令示例：`wsl -e bash -c "source ~/miniconda3/bin/activate freqtrade-gpu && python script.py"`

4. **GPU Environment** / GPU环境
   - GPU型号：NVIDIA GeForce RTX 3060 Ti (8GB)
   - CUDA版本：12.5
   - cuDF版本：24.12.00
   - 验证GPU：`wsl -e bash -c "nvidia-smi"`

5. **File Path Conversion** / 文件路径转换
   - Windows路径：`D:\project\freqtrade-strategies`
   - WSL2路径：`/mnt/d/project/freqtrade-strategies`
   - 在WSL2中访问Windows文件使用 `/mnt/` 前缀

6. **Common WSL2 Commands** / 常用WSL2命令
   - 检查WSL状态：`wsl --status`
   - 重启WSL：`wsl --shutdown` (在PowerShell中)
   - 列出分发版：`wsl --list --verbose`
   - 进入WSL：`wsl` (默认分发版)

## Project Management Rules / 项目管理规则

1. **Kanban Board Management** / 看板管理
   - 使用 `docs/KANBAN.md` 进行任务跟踪和项目管理
   - 遵循标准的Kanban工作流：待办 → 进行中 → 待审查 → 已完成
   - 开始、暂停或完成工作时立即更新任务状态
   - 每日检查和更新看板

2. **Task Numbering Convention** / 任务编号规范
   - 使用格式 `[TASK-XXX]`，其中XXX为三位数字
   - 编号范围：
     - `001-099`: 性能优化任务
     - `100-199`: 基础设施和部署任务
     - `200-299`: 策略开发任务
     - `300-399`: 回测工具相关任务
     - `400-499`: 测试和质量保证任务
     - `500-599`: 文档和教程任务

3. **Task Description Standards** / 任务描述标准
   - 包含任务编号、标题和优先级
   - 对于复杂任务，提供详细描述：
     - 目标：要达成的具体目标
     - 范围：涉及的模块和文件
     - 预期收益：性能提升、功能改进等
     - 技术栈：使用的技术和工具
     - 验证方法：如何验证任务完成

4. **Priority Management** / 优先级管理
   - 高优先级：核心功能、性能瓶颈、阻塞性问题
   - 中优先级：重要功能增强、优化改进
   - 低优先级：辅助功能、文档完善、长期规划
   - 除非被阻塞，否则始终优先处理高优先级任务

5. **Task Workflow** / 任务工作流
   - 创建任务时：添加到待办区域，分配编号和优先级
   - 开始工作时：移至进行中区域
   - 代码就绪时：移至待审查区域进行代码审查和测试
   - 完成时：移至已完成区域并记录完成日期
   - 限制进行中任务数量以避免上下文切换（每人最多2-3个任务）