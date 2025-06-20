


# LLM-Dev-Cookbook

![Project Banner](https://placehold.co/1200x400/000000/FFFFFF/png?text=LLM-Dev-Cookbook)

这是一个帮助开发者快速入门LLM应用开发的学习笔记集合。


[🛄教程汇总](https://github.com/tataCrayon/LLM-DEV-COOKBOOK/blob/main/doc/Github%E9%A1%B9%E7%9B%AE%E4%B8%8E%E6%95%99%E7%A8%8B%E6%B1%87%E6%80%BB.md)

---

## 🚀 项目特色

*   **📚 从0~1循序渐进**: 代码示例从Python基础、RAG核心组件到开发框架。
*   **🛠️ 开箱即练**: 所有example均有完善注释，边学边练。
*   **🌱 持续更新**: 目标是入门到进阶，持续更新。

---

## 📂 项目结构

本项目结构如下：
```txt
├── .vscode/ # VSCode 编辑器配置
├── data/ # 存放原始数据文件 (如 .pdf, .txt)
│
├── doc/ # 存放学习资料、设计文档、参考链接等 !!!CORE!!!
│
├── fast_langchain_example/ 
│ ├── chains/ # 业务逻辑链的具体实现
│ ├── core/ # 应用核心组件 (加载器、切分器、嵌入等)
│ └── prompt/ # 提示词模板管理
├── fast_python_example/ # 基础: LLM开发相关的Python基础知识示例
│ ├── base_python/
│ └── llm_python/
├── fast_rag_example/ # 示例: RAG各组件的独立、可运行示例
│ ├── example_chromadb.py
│ ├── example_embedding_zh.py
│ └── ...
├── scripts/ # 存放独立的工具脚本 (如数据下载、预处理)
├── tests/ # 自动化测试用例
├── .env # 环境变量示例文件 (请复制为.env并填入您的密钥)
├── .gitignore # Git忽略文件配置
└── README.md
```

- fast_langchain_example  
相较官网示例，本项目example组件联系更密接，有关联承接。  
deepseek_chat.py:使用LangChain构建DeepSeek+联网搜索的Agent。  
其他：LangChain各组件example  

- fast_python_example  
LLM开发相关的Python基础知识示例  

- fast_rag_example  
RAG各组件的独立、可运行示例
---

## ⚙️ 如何开始

请遵循以下步骤来配置和运行本项目。

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
```
### 2. 创建并激活虚拟环境（可选，想快速过一遍则跳过）
建议使用虚拟环境来隔离项目依赖。
```
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
项目所需的所有库都记录在 requirements.txt 文件中。
```bash
pip install -r requirements.txt
```
### 4. 配置环境变量

您需要一个地方存放API密钥等敏感信息，即创建一个`.env`文件。

```bash
# .env示例文件
DEEPSEEK_API_KEY="sk-xxx"
SERPAPI_API_KEY="xxx"
```

## 📖 如何使用 (学习路径建议)

建议您按照以下顺序来探索这个项目：

Python基础 (fast_python_example): 学习LLM应用开发必须了解的Python知识。

RAG组件 (fast_rag_example): 这是项目的核心亮点。您可以逐个运行此目录下的example_*.py文件，以理解RAG（检索增强生成）的每个关键部分是如何工作的。

完整应用 (fast_langchain_example): 当您理解了各个组件后，可以研究这个目录。它向您展示了如何将所有独立的组件有机地组织起来，构建成一个完整的、可维护的聊天机器人或RAG应用。

文档 (doc): 此目录存放了为您整理的LLM相关的学习资料，可以作为您深入学习的参考。

# ✨学习方法总结

- **公开学习，输出倒逼输入。**
- **借鉴别人的学习路线与笔记，结合书籍，多看优秀的Github项目。**
- **将自己学习、整理的思路形成Prompt，以让大模型辅助回答整理格式。**
- **和代码片段库一样，要有自己的Prompt库。**
- **问题主导。学完后找一些常见QA回答，学的时候也带着问题去学。**
- **知识面有广度后，阅读源码，尝试手搓代码。**
- **简化主干，尝试一句话总结复杂描述。**
- **定期计划梳理，多个事项长期并行有点累人，可以做专项突破。**
- **学习完一阶段的知识后，可以去看一下面试题做补充。**