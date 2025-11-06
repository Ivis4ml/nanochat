# nanochat Deep Dive 系列

> 深入解析 nanochat 的每一个核心模块和脚本

## 📚 系列概述

本系列文档深入剖析 nanochat 项目的实现细节，每篇文档聚焦一个模块或脚本，从设计理念、代码实现到实际应用全方位讲解。

## 🗺️ 学习路线图

### 第一部分：基础设施（已完成）

这些模块提供了整个项目的基础功能：

- ✅ [01. configurator.py](01_configurator.md) - **配置系统**
  - 极简配置系统，支持配置文件和命令行参数
  - "Poor Man's Configurator" 设计哲学

- ✅ [02. common.py](02_common.md) - **通用工具库**
  - 项目全局配置和工具函数
  - 设备管理、目录管理、分布式训练支持

- ✅ [03. dataset.py](03_dataset.md) - **数据集加载**
  - FineWeb-Edu 100B 数据集下载和迭代
  - Parquet 格式、流式处理、DDP 支持

- ✅ [04. tokenizer.py](04_tokenizer.md) - **BPE 分词器**
  - GPT-4 风格的 BPE 实现
  - HuggingFace 和 RustBPE+tiktoken 双实现
  - 对话渲染机制

### 第二部分：数据处理（进行中）

将原始文本转换为模型训练所需的格式：

- 🔄 [05. dataloader.py](05_dataloader.md) - **数据加载器**（下一篇）
  - 文本打包成固定长度序列
  - 批处理和分布式数据并行
  - 内存映射和高效采样

- ⏸️ [06. tok_train.py](06_tok_train.py.md) - **分词器训练脚本**
  - 如何使用 dataset 训练 tokenizer
  - Token bytes 缓存生成

### 第三部分：模型架构

- ⏸️ [07. gpt.py](07_gpt.md) - **GPT 模型实现**
  - Transformer 架构详解
  - 注意力机制、MLP、Layer Norm
  - 模型初始化策略

### 第四部分：优化器

- ⏸️ [08. adamw.py](08_adamw.md) - **AdamW 优化器**
  - PyTorch 原生实现
  - 权重衰减和学习率调度

- ⏸️ [09. muon.py](09_muon.md) - **Muon 优化器**
  - 新型优化器实现
  - 与 AdamW 的对比

### 第五部分：训练引擎

- ⏸️ [10. engine.py](10_engine.md) - **训练引擎**
  - 训练循环核心逻辑
  - 梯度累积、混合精度训练
  - 检查点保存和恢复

- ⏸️ [11. checkpoint_manager.py](11_checkpoint_manager.md) - **检查点管理**
  - 模型保存和加载策略
  - 分布式检查点处理

### 第六部分：预训练

- ⏸️ [12. base_train.py](12_base_train.md) - **预训练脚本**
  - 完整的预训练流程
  - 超参数配置和调优

- ⏸️ [13. base_eval.py](13_base_eval.md) - **预训练评估**
  - 验证集损失计算
  - 困惑度（Perplexity）评估

- ⏸️ [14. mid_train.py](14_mid_train.md) - **中期训练**
  - 在更多数据上继续训练
  - 学习率调整策略

### 第七部分：微调（SFT）

- ⏸️ [15. chat_sft.py](15_chat_sft.md) - **监督微调**
  - 对话格式训练
  - 掩码损失策略

### 第八部分：强化学习（RL）

- ⏸️ [16. chat_rl.py](16_chat_rl.md) - **强化学习微调**
  - RLHF 实现
  - 奖励模型和策略优化

### 第九部分：评估

- ⏸️ [17. core_eval.py](17_core_eval.md) - **CORE 基准评估**
  - 多项选择题评估
  - MMLU、ARC、HumanEval 等

- ⏸️ [18. loss_eval.py](18_loss_eval.md) - **损失评估**
  - Bits per byte 计算
  - 验证集指标

- ⏸️ [19. chat_eval.py](19_chat_eval.md) - **对话评估**
  - ChatCORE 基准
  - 对话质量评估

### 第十部分：推理和服务

- ⏸️ [20. chat_cli.py](20_chat_cli.md) - **命令行推理**
  - 交互式对话界面
  - 文本生成和采样

- ⏸️ [21. chat_web.py](21_chat_web.md) - **Web 服务**
  - FastAPI 实现
  - ChatGPT 风格的 UI

### 第十一部分：辅助工具

- ⏸️ [22. report.py](22_report.md) - **报告生成**
  - Markdown 格式的训练报告
  - 指标可视化

- ⏸️ [23. execution.py](23_execution.md) - **执行控制**
  - 进程管理和监控
  - 错误处理和日志

## 📖 阅读建议

### 新手路线（快速入门）

如果你刚接触 nanochat，建议按以下顺序阅读：

1. **01. configurator.py** - 了解配置系统
2. **04. tokenizer.py** - 理解文本如何转换为 token
3. **03. dataset.py** - 了解数据来源
4. **07. gpt.py** - 理解模型架构
5. **12. base_train.py** - 看完整训练流程

### 深度学习路线（系统学习）

如果你想深入理解整个系统，建议按编号顺序阅读：

**阶段 1**：基础（01-06）→ 理解数据处理流程
**阶段 2**：模型（07-11）→ 理解训练核心
**阶段 3**：训练（12-16）→ 理解完整训练流程
**阶段 4**：评估（17-19）→ 理解模型评估
**阶段 5**：应用（20-23）→ 理解推理和部署

### 实战路线（动手实践）

如果你想快速上手修改和实验：

1. **运行 speedrun.sh** - 先跑通整个流程
2. **03. dataset.py** - 了解如何准备自己的数据
3. **04. tokenizer.py** - 训练自定义 tokenizer
4. **12. base_train.py** - 修改训练参数
5. **20. chat_cli.py** - 与模型交互

## 🎯 每篇文档的结构

每篇 deep dive 文档都包含：

- **概述**：模块的作用和在项目中的位置
- **核心概念**：必要的背景知识
- **代码详解**：逐函数/逐类分析
- **设计亮点**：优秀的设计决策
- **使用示例**：实际应用代码
- **常见问题**：Q&A 和故障排查
- **性能分析**：性能考虑和优化建议
- **相关模块**：与其他模块的关系

## 💡 文档约定

- ✅ 已完成
- 🔄 进行中
- ⏸️ 待完成
- 🔍 需要更新

## 🤝 贡献

欢迎贡献改进建议！如果你发现：
- 文档有错误或不清楚的地方
- 缺少重要的实现细节
- 有更好的解释方式

请提交 PR 或 Issue。

## 📊 更新日志

- **2025-11-06**:
  - ✅ 完成 01-04（基础设施部分）
  - 🔄 正在编写 05（dataloader.py）
  - 📝 规划完整系列结构（23 篇）

---

**下一步**: 阅读 [05. dataloader.py](05_dataloader.md) 了解数据批处理机制
