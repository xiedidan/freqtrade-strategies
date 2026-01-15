---
inclusion: always
---

# Development Rules / 开发规则

## Language and Documentation Rules / 语言和文档规则

1. **Think in English, code in English** / 用英文思考，用英文编程
   - Use English for all code comments and configuration file comments
   - Use English for variable names, function names, and class names
   - Use English for commit messages and technical documentation

2. **Respond in Chinese** / 用中文回答
   - All responses to users should be in Chinese
   - All user-facing documentation should be in Chinese
   - Error messages and logs can remain in English for technical clarity

## Environment Management Rules / 环境管理规则

1. **Use venv for Python environment management** / 使用venv进行Python环境管理
   - Always use Python virtual environments (venv) for development
   - Create virtual environment: `python -m venv venv`
   - Activate virtual environment: 
     - Windows: `venv\Scripts\activate`
     - Linux/Mac: `source venv/bin/activate`
   - Install dependencies within the virtual environment
   - Keep requirements.txt updated with all dependencies

2. **Custom Dependencies Management** / 自定义依赖管理
   - Use `requirements-custom.txt` for project-specific dependencies
   - This file contains dependencies for:
     - Parallel backtest tool (tqdm for progress bars)
     - Testing frameworks (hypothesis for property-based testing, pytest)
   - Install custom dependencies: `pip install -r requirements-custom.txt`
   - Keep this file separate from Freqtrade's main requirements.txt
   - Update this file when adding new custom tools or testing dependencies

## Code Quality Standards / 代码质量标准

- Follow PEP 8 style guidelines for Python code
- Use meaningful English names for variables and functions
- Add comprehensive English comments for complex logic
- Maintain clean and readable code structure