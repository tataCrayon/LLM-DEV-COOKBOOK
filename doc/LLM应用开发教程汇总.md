# LLM应用开发教程汇总

致力于帮助你我成为LLM应用开发工程师的教程汇总。  
>网上优秀的视频特别多，只是我不爱看视频，所以视频资源列举的比较少。

---

## 1️⃣Python

主要了解Python基础语法，常见库类。  

### 基础语法  

- [Python3官方教程（英文）](https://docs.python.org/3/tutorial/)
- [Python3官方教程（中文）](https://docs.python.org/zh-cn/3/tutorial/)
- [菜鸟教程](https://www.runoob.com/python/python-tutorial.html)：基础教程。
- [codeDex](https://www.codedex.io/python)：趣味学Python的网站。
- [realpython](https://realpython.com/)：高质量Python教程。


### 常见库类  

熟悉用于生成式AI的关键Python库包括TensorFlow、PyTorch、Hugging Face Transformers、Diffusers、Jax、LangChain、LlamaIndex和Weight and Biases 。还有数据科学库Numpy和Pandas。
这些库通过提供预编写的代码模块和优化的算法，极大地简化了复杂的编程任务，从而加快了开发速度并减少了错误 。

-  **数据科学与 AI 核心库**
    - NumPy: 数值计算, 数组操作
        - 官方文档: [https://numpy.org/doc/stable/user/index.html](https://numpy.org/doc/stable/user/index.html)
        - 中文文档: [https://www.numpy.org.cn/user/index.html](https://www.numpy.org.cn/user/index.html)
        - 菜鸟教程: [https://www.runoob.com/numpy/numpy-tutorial.html](https://www.runoob.com/numpy/numpy-tutorial.html)
    - Pandas: 数据分析与处理, DataFrame
        - 官方文档: [https://pandas.pydata.org/docs/user_guide/index.html](https://pandas.pydata.org/docs/user_guide/index.html)
        - 中文文档 (社区): [https://www.pypandas.cn/docs/](https://www.pypandas.cn/docs/)
        - 10 Minutes to pandas: [https://pandas.pydata.org/docs/user_guide/10min.html](https://pandas.pydata.org/docs/user_guide/10min.html)
    - Matplotlib & Seaborn: 数据可视化
        - Matplotlib 官方教程: [https://matplotlib.org/stable/tutorials/index.html](https://matplotlib.org/stable/tutorials/index.html)
        - Seaborn 官方教程: [https://seaborn.pydata.org/tutorial.html](https://seaborn.pydata.org/tutorial.html)
        - Matplotlib 中文文档: [https://www.matplotlib.org.cn/tutorials/](https://www.matplotlib.org.cn/tutorials/)
    - Scikit-learn: 核心机器学习库 (分类, 回归, 聚类, 模型评估, 预处理)
        - 官方文档: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
        - 中文文档 (社区): [https://scikit-learn.org.cn/view/6.html](https://scikit-learn.org.cn/view/6.html)

### 建议  

可以分阶段学习，先了解python基础语法、数据科学库（Numpy和Pandas）基本用法，在学RAG和Agent知识时边练习边补充。


### 深入学习资料  

*   **[FastAPI官方文档](https://fastapi.tiangolo.com/)**
    *   **简介**: FastAPI 是一个现代、快速（高性能）的 Web 框架，用于使用 Python 构建 API，基于标准 Python 类型提示。


*   **[NLP 精通指南：从零到英雄使用](https://codanics.com/nlp_huggingface_guide/)**
    *   **简介**: 自然语言处理（NLP）是人类语言与计算机理解之间的桥梁。无论您是想构建聊天机器人、分析情感、翻译语言，还是创造 AI 的下一个突破，这本全面的指南都将带您从绝对初学者成长为高级从业者。


*   **[Hugging Face 101：绝对初学者的教程！](https://dev.to/pavanbelagatti/hugging-face-101-a-tutorial-for-absolute-beginners-3b0l)**


*   **[TensorFlow官网教程](https://www.tensorflow.org/learn?hl=zh-cn)**
    *   **简介**: 借助 TensorFlow，初学者和专家可以轻松创建适用于桌面、移动、Web 和云环境的机器学习模型。


*   **[PyTorch官网文档](https://docs.pytorch.org/docs/stable/index.html)**
    *   **简介**: PyTorch 是一个用于深度学习的优化张量库，支持 GPU 和 CPU。
    *   **项目**: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。一个在 PyTorch 中从零开始实现类似 ChatGPT 的 LLM 的项目。


---

## 2️⃣LLM基础  

资料很多，这里列举两个。


- https://github.com/ZJU-LLMs/Foundations-of-LLMs  
旨在为对大语言模型感兴趣的读者系统地讲解相关基础知识、介绍前沿技术。  
作者团队在B站有视频：[浙江大学-大模型原理与技术](https://www.bilibili.com/video/BV1PB6XYFET2)


- https://github.com/datawhalechina/so-large-lm  （推荐）
旨在作为一个大规模预训练语言模型的教程，从数据准备、模型构建、训练策略到模型评估与改进，以及模型在安全、隐私、环境和法律道德方面的方面来提供开源知识。

### 扩展学习  

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762):Transformer的开山之作。


### 深入学习

在初步完成LLM应用开发学习后，进行LLM基础深入学习。

*   **[《动手学深度学习》](https://github.com/d2l-ai/d2l-zh)**
    *   **简介**: 由亚马逊科学家团队撰写的权威入门教程，提供从零开始的深度学习理论和代码实践。是理解后续一切模型的基础。

*   **[Deep Learning Book (Chapter 6-8)](https://www.deeplearningbook.org/)**
    *   **简介**: 第六、七、八章主要涵盖了深度前馈网络的基础知识、训练中的常见问题及其解决方案，以及优化算法。

*   **[Hugging Face NLP 课程 (中文版)](https://huggingface.co/learn/nlp-course/zh-CN/chapter1/1)**
    *   **简介**: Hugging Face官方教程，带你学习如何使用 Transformers 库解决NLP问题，是连接理论与实践的最佳桥梁。

---

## 3️⃣Prompt

首推⬇️：

提示词工程 (Prompt Engineering)
*   **[《提示工程指南》](https://www.promptingguide.ai/zh)**
    *   **简介**: 一个极其全面的提示工程学习网站，覆盖从基础技巧到CoT（思维链）等前沿技术。
    *   **定位**: **核心方法论**。学习如何与LLM高效“沟通”。

*   **[吴恩达《面向开发者的提示工程》](https://github.com/datawhalechina/llm-cookbook)**
    *   **简介**: 项目是一个面向开发者的大模型手册，针对国内开发者的实际需求，主打 LLM 全方位入门实践。本项目基于吴恩达老师大模型系列课程内容，对原课程内容进行筛选、翻译、复现和调优，覆盖从 Prompt Engineering 到 RAG 开发、模型微调的全部流程，用最适合国内学习者的方式，指导国内开发者如何学习、入门 LLM 相关项目。
    *   **定位**: 用最短的时间掌握最核心的Prompt技巧。

### 扩展学习  

来自LangChain团队博客的[《communication-is-all-you-need》](https://blog.langchain.com/communication-is-all-you-need/)。


*   **[Prompt 越狱攻击技术手册](https://github.com/Acmesec/PromptJailbreakManual)**
    *   **简介**: 由米斯特安全团队维护，收录了大量提示词越狱攻击的案例与技术。


---

## 4️⃣RAG

RAG（Retrieval-Augmented Generation）是 LLM 应用最重要的"事实落地"手段。从 Naive RAG 到 Agentic RAG，工程深度堪比 Agent 系列——分块、检索、重排、评估每一层都是独立学科。
> 详细的 RAG 系列学习文章正在筹备中，建成后将放在 [doc/rag/](./rag/) 目录。

### 核心必读（按优先级排序）

- **[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)**（推荐首读）
  同济大学综述，提出 **Naive RAG → Advanced RAG → Modular RAG** 三代演进模型，是后续所有 RAG 论文的引用起点。

- **[Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)**
  Anthropic 提出的"上下文检索"技术：在 chunk 前预置 LLM 生成的上下文摘要，检索失败率下降 49%（叠加 rerank 后 67%）。生产级 RAG 必看。

- **[LangChain: RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)**
  LangChain 团队的 14 集系列视频，从 indexing → retrieval → generation 每个环节拆解，工程师入门首选。
  - 中文版：[大佬"沧海九粟"《从零开始学习 RAG》](https://www.bilibili.com/video/BV1dm41127jc/)

- **[NVIDIA: What Is Retrieval-Augmented Generation?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)**
  RAG 概念发明人之一（Patrick Lewis）所在团队的官方科普，配图清晰，适合给业务方讲解。

- **[Microsoft: Azure AI Search RAG Pattern Guide](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)**
  企业级 RAG 方案设计与评估完整指南，涵盖文档预处理、分块、混合检索、评估闭环。

### RAG 范式演进

- **[Lewis et al.: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)](https://arxiv.org/abs/2005.11401)**：RAG 概念原始论文（NeurIPS 2020），定义了 retriever + generator 端到端联合训练范式。
- **[Modular RAG: Transforming RAG Systems into LEGO-like Frameworks](https://arxiv.org/abs/2407.21059)**：Modular RAG 系统性归纳——把 RAG 拆成 Indexing / Pre-Retrieval / Retrieval / Post-Retrieval / Generation / Orchestration 六大模块。
- **[Awesome-LLM-RAG (jxzhangjhu)](https://github.com/jxzhangjhu/Awesome-LLM-RAG)**：RAG 相关论文、教程、博客的 awesome 清单（持续更新）。
- **[awesome-rag (frutik)](https://github.com/frutik/Awesome-RAG)**：另一份高质量 RAG 资源汇总，按主题分类。

### 文档处理与分块（Chunking）

- **[Pinecone: Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)**：分块策略全景——fixed-size / recursive / semantic / document-specific，每种的取舍。
- **[Unstructured](https://docs.unstructured.io/welcome)**：开源文档解析工具，原生支持 PDF/DOCX/PPT/HTML 等 20+ 格式，输出统一的 Element 结构。
- **[Docling (IBM)](https://github.com/docling-project/docling)**：IBM 开源的文档解析引擎，强项是表格识别和 layout-aware 解析。
- **[LlamaParse](https://docs.cloud.llamaindex.ai/llamaparse/getting_started)**：LlamaIndex 出品，专为复杂文档（含表格/公式）设计的解析服务，对 PDF 表格支持优秀。
- **[Nougat (Meta)](https://github.com/facebookresearch/nougat)**：学术论文专用 OCR，能把公式还原为 LaTeX。
- **[Late Chunking](https://weaviate.io/blog/late-chunking)**：Weaviate 提出的"先编码整文档再分块"技术，保留上下文信息，召回质量明显提升。

### Embedding 与检索

- **[MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)**：Hugging Face 维护的多任务 Embedding 排行榜，选模型的事实标准。
- **[BGE 系列 (BAAI)](https://github.com/FlagOpen/FlagEmbedding)**：智源开源的中英文双语 embedding，BGE-M3 同时支持 dense / sparse / multi-vector。
- **[ColBERT v2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)**：late interaction 范式，比 dense embedding 更细粒度。
- **[Stanford ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)**：跳过 OCR，用 VLM 直接对文档图像做 late interaction 检索，多模态 RAG 新基线。
- **[Cohere Rerank](https://cohere.com/rerank)**：商用 cross-encoder 重排服务，对召回结果二次排序提升精度。
- **[Hybrid Search Explained (Pinecone)](https://www.pinecone.io/learn/hybrid-search-intro/)**：dense + sparse（BM25）混合检索原理与实现。
- **[HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)**：先让 LLM"假设"答案再用假设去检索的 query 改写技巧。

### 向量数据库选型

- **[Milvus](https://milvus.io/docs)**：分布式架构，亿级向量首选，2.5 版本支持稀疏向量、全文检索、混合搜索。
- **[Qdrant](https://qdrant.tech/documentation/)**：Rust 编写，开发者体验优秀，filter + 向量混合查询性能强。
- **[Weaviate](https://weaviate.io/developers/weaviate)**：内置混合检索 + 模块化向量化，BYOV（bring your own vectorizer）友好。
- **[pgvector](https://github.com/pgvector/pgvector)**：PostgreSQL 扩展，适合"已经在用 Postgres"的团队，0.7+ 支持 halfvec 和 HNSW 性能跃升。
- **[Chroma](https://docs.trychroma.com/)**：轻量本地优先，适合 prototype 和小规模生产。
- **[LanceDB](https://lancedb.github.io/lancedb/)**：嵌入式、多模态、列式存储，适合"数据湖+向量"统一查询。
- **[Vector Database Benchmark (VectorDBBench)](https://github.com/zilliztech/VectorDBBench)**：Zilliz 维护的开源向量库 benchmark，可复现选型对比。

### 高级范式

- **[Microsoft GraphRAG](https://microsoft.github.io/graphrag/)**：基于知识图谱 + 社区检测的 RAG 范式，擅长"全局问题"（如总结主题、关联实体）。
  - 论文：[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- **[Neo4j GraphRAG Manifesto](https://neo4j.com/blog/graphrag-manifesto/)**：Neo4j 视角的 GraphRAG 实现路径与最佳实践。
- **[Agentic RAG: A Survey on Agentic Retrieval-Augmented Generation](https://arxiv.org/abs/2501.09136)**：Agentic RAG 综述，把 Agent 决策能力引入 RAG 检索流程。
  - 入门博客：[ByteByteGo: RAG vs Agentic RAG](https://blog.bytebytego.com/p/ep169-rag-vs-agentic-rag)
- **[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)**：让模型自己决定"何时检索"和"是否信任检索结果"。
- **[Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)**：检索结果质量评估 + 失败时回退到网络搜索的纠错范式。
- **[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)**：递归聚类生成层次化摘要树，多粒度检索利器。
- **[HippoRAG: Neurobiologically Inspired Long-Term Memory](https://arxiv.org/abs/2405.14831)**：受海马体启发的图谱检索范式，多跳问题性能大幅提升。

### 评估与可观测

- **[RAGAS](https://docs.ragas.io/)**：开源 RAG 评估框架，提供 faithfulness / answer relevancy / context precision / context recall 等核心指标。
- **[TruLens](https://www.trulens.org/)**：RAG 三角评估（Context Relevance / Groundedness / Answer Relevance）的开源实现。
- **[ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476)**：自动化 RAG 评估框架，基于 LLM-as-judge 但有统计校准。
- **[DeepEval](https://docs.confident-ai.com/)**：pytest 风格的 LLM/RAG 单元测试框架，CI 友好。
- **[LangSmith](https://docs.smith.langchain.com/)**：LangChain 官方追踪与评估平台，RAG pipeline 调试首选。

### 生产化：长上下文、缓存、新鲜度

- **[Anthropic: Long Context vs RAG](https://www.anthropic.com/engineering/contextual-retrieval)**：长上下文模型时代 RAG 是否还需要？Anthropic 的答案是"互补"。
- **[LlamaIndex: How to Build a Production-Ready RAG System](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)**：LlamaIndex 官方生产级 RAG 优化清单。
- **[Towards Long Context RAG (LlamaIndex)](https://www.llamaindex.ai/blog/towards-long-context-rag)**：长上下文条件下的 RAG 设计模式。
- **[OpenAI: Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)** / **[Anthropic: Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)**：缓存检索到的上下文，降低成本和首字延迟。

### 速成与入门

- 视频：[LangChain RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) / 中文版 [《从零开始学习 RAG》](https://www.bilibili.com/video/BV1dm41127jc/)
- 课程：[DeepLearning.AI: Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/)（吴恩达 + LlamaIndex）
- 课程：[DeepLearning.AI: Knowledge Graphs for RAG](https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/)（吴恩达 + Neo4j）
- 教程：[Pinecone Learning Center](https://www.pinecone.io/learn/)：从向量基础到 RAG 工程的系统教程。

### 扩展阅读

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)：长上下文中的"中间塌陷"现象——重排和上下文压缩的理论依据。
- [DRAGIN: Dynamic Retrieval Augmented Generation](https://arxiv.org/abs/2403.10081)：动态决定何时触发检索的范式。
- [Active Retrieval Augmented Generation (FLARE)](https://arxiv.org/abs/2305.06983)：生成过程中主动触发检索的实现。
- [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture](https://arxiv.org/abs/2401.08406)：微软的 RAG vs Fine-tuning 实证研究。

---

## 5️⃣Agent

Agent 是当前 LLM 应用开发最重要的方向。从单次调用到自治系统，需要掌握的工程知识远超”会调 API”。  
> 详细的 Agent 系列学习文章见 [doc/agent/](./agent/) 目录。

### 核心必读（按优先级排序）

- **[Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)**（推荐首读）  
  Anthropic 官方的 Agent 构建指南，讲透了何时用 workflow 何时用 agent、常见架构模式（prompt chaining、routing、orchestrator-worker、evaluator-optimizer）以及”从简单开始”的核心原则。

- **[Martin Fowler / Böckeler: Harness Engineering for Coding Agent Users](https://martinfowler.com/articles/harness-engineering.html)**  
  Agent = Model + Harness。系统讲解 Guides/Sensors（前馈/反馈）、三层调控（Maintainability / Architecture / Behaviour）、Computational vs Inferential、Ashby's Law 在 Agent 中的应用。

- **[OpenAI: Harness Engineering — Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/)**  
  OpenAI 对 Harness 概念的官方阐释，重点在 Codex 的 Loop 设计。

- **[LangChain: State of Agent Engineering Report](https://www.langchain.com/state-of-agent-engineering)**  
  2025 年行业调查报告，覆盖 Agent 采用率、部署模式、失败原因、最佳实践。57% 企业已上线 Agent。

- **[Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)**（推荐）  
  经典综述，将 Agent 拆解为 Planning + Memory + Tool Use，虽发表于 2023 但思路框架仍然适用。

### Agent 基础范式

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io/)：ReAct 框架原始论文。
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)：Agent 通过语言反思自我改进。
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)：将思维链扩展为思维树，允许搜索与回溯。
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)：先规划后执行的范式。

### Harness Engineering & Loop Engineering

- **[boydfd: 拆解 Harness Engineering 和 Loop Engineering](https://www.cnblogs.com/boydfd/p/20525224)**  
  五层架构拆解（Cross-cutting → Framework → Design Axes → Pattern → Instance）、10 个 Design Axes、Mechanism vs Policy。国内最深度的中文解读。

- **[puppyone: Loop Engineering — 5 Building Blocks + The Missing One](https://www.puppyone.ai/en/blog/what-is-loop-engineering-5-building-blocks-missing-one)**  
  Loop 的 5 个构件（Automations、Worktrees、Skills、Connectors、Sub-agents）+ 第 6 块 Workspace（Identity/Scope/Audit/Rollback）。

- **[Data Science Dojo: Agentic Loops Explained — From ReAct to Loop Engineering](https://datasciencedojo.com/blog/agentic-loops-explained-from-react-to-loop-engineering-2026-guide/)**  
  从 ReAct 到 Loop Engineering 的全景演进指南。

### Multi-Agent 系统

- **[Azure Architecture Center: AI Agent Orchestration Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)**  
  微软官方的 Agent 编排模式参考架构。

- **[LangChain: Choosing the Right Multi-Agent Architecture](https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture)**  
  Multi-Agent 架构选型指南。

- **[Confluent: Four Design Patterns for Event-Driven Multi-Agent Systems](https://www.confluent.io/blog/event-driven-multi-agent-systems/)**  
  事件驱动的 Multi-Agent 设计模式。

- **[RUCAIBox: awesome-agent-harness](https://github.com/RUCAIBox/awesome-agent-harness)**  
  Agent Harness 相关论文和资源的 awesome list。

### MCP（Model Context Protocol）

- **[MCP 官方规范](https://modelcontextprotocol.io/specification/2025-03-26)**  
  Anthropic 主导的 Model Context Protocol 完整技术规范。

- **[Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)**  
  MCP 的设计动机与愿景。

- **[Anthropic: Writing Effective Tools for AI Agents](https://www.anthropic.com/engineering/writing-tools-for-agents)**  
  如何设计好的工具 schema，让 Agent 能正确、高效地使用工具。

- **[MCP Spec 2025-06-18 更新](https://forgecode.dev/blog/mcp-spec-updates/)**  
  最新规范变更：安全增强、Structured Output、Elicitation 等。

### 安全、护栏与评估

- **[OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)**  
  LLM 应用安全十大风险，Agent 场景尤其需要关注 prompt injection、insecure output handling、excessive agency。

- **[JetBrains: LLM Evaluation and AI Observability for Agent Monitoring](https://blog.jetbrains.com/pycharm/2026/05/llm-evaluation-and-ai-observability-for-agent-monitoring/)**  
  Agent 评估与可观测性实践指南。

- **[AI Agent Guardrails: Production Enterprise Safety Guide](https://devops.gheware.com/blog/posts/ai-agent-guardrails-production-enterprise-2026.html)**  
  企业级 Agent 护栏部署完整指南。

### 速成与入门

- [速成课：使用开源工具构建 AI 代理](https://github.com/patchy631/ai-engineering-hub/tree/main/agent-with-mcp-memory)：讲”什么是 AI Agent”、工具连接、MCP 替换、可观察性。
- [roadmap.sh: AI Agents](https://roadmap.sh/ai-agents)：社区驱动的 AI Agent 学习路线图。
- 书籍：《大模型应用开发 动手做AI Agent》——入门实操。

### 扩展阅读

- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)：解释长上下文处理的挑战，”为什么要把任务拆分”的理论依据。
- [Google A2A: Agent-to-Agent Protocol](https://google.github.io/A2A/)：Agent 间通信的开放协议。
- [HumanLayer](https://humanlayer.dev/)：”It's not a model problem. It's a configuration problem.” 人机协作层。
---

## 6️⃣框架

当前框架百花齐放，这里列举两个听到的最多的。

*   **[LangChain 官方文档](https://python.langchain.com/docs/introduction/)**
    *   **简介**: LangChain是当下最流行的LLM应用开发框架。
    *   **定位**: 通用AI应用编排框架。**生态最广**、功能最全、Agent能力最成熟。

*   **[LlamaIndex 官方文档](https://docs.llamaindex.ai/en/stable/)**
    *   **简介**: LlamaIndex 是构建基于 LLM 的数据代理的领先框架，使用 LLM 和工作流。
    *   **定位**: 以数据为中心的RAG框架。**RAG能力最深**、数据交互策略最丰富，Agent能力相对较弱、生态广度不及LangChain

框架非常多，各有特长。还有一些专于监控、评估的。

---

## 7️⃣项目

*   **[awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)**  
精选收录了采用RAG、AI智能体、多智能体团队、MCP、语音智能体等技术构建的超赞LLM应用。

*   **[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)**
    *   **简介**: 一个现象级的开源项目，提供了完整的、基于本地知识库的问答解决方案。是学习企业级RAG应用架构的最佳案例。
    *   **定位**: **RAG最佳实践**。从代码层面深入理解RAG的全流程。

*   **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**
    *   **简介**: 一个简单易用的、集成了多种高效微调方案的LLM微调框架。让开发者能用消费级显卡训练自己的模型。
    *   **定位**: **微调实战**。当你需要模型掌握特定领域知识或风格时，这是必经之路。

*   **[MetaGPT](https://github.com/geekan/MetaGPT)**
    *   **简介**: 一个惊艳的多智能体框架，输入一句话需求即可模拟软件公司流程并生成完整代码。
    *   **定位**: **Agent前沿**。理解多智能体协作的设计思想与工程实现。

*   **[Lobe Chat](https://github.com/lobehub/lobe-chat)**
    *   **简介**: 一个开源的、高性能的聊天机器人框架，UI精美，功能强大，支持插件系统。可以直接部署使用，也是学习如何构建一个完整、高质量LLM应用的绝佳范例。
    *   **定位**: **动手实践**。通过研究这个项目的代码，你可以学到全栈LLM应用的工程实践。

---

## 8️⃣八股  

*   **[LLM 面试问题汇总 (LLM-Interview-Questions)](https://github.com/wdndev/llm_interview_note)**
    *   **简介**: 一个由社区共同维护的、非常活跃的LLM面试题库。它不仅有题目，还有社区成员提供的参考答案和讨论。
    *   **定位**: **求职宝典**。在面试前刷一遍，能极大地提升你的信心和成功率。


---


## 9️⃣其它优秀教程

*   **[happy-llm](https://github.com/datawhalechina/happy-llm)**
    *   **简介**: 📚 从零开始的大语言模型原理与实践教程。深入理解 LLM 核心原理，动手实现你的第一个大模型。

*   **[谷歌机器学习教程](https://developers.google.com/machine-learning?hl=zh-cn)**
    *   **简介**: 通过 Google 的机器学习课程学习如何构建机器学习产品。

*   **[LLM 知识汇总(LLMForEverybody)](https://github.com/luhengshiwo/LLMForEverybody)**
    *   **简介**: 每个人都能看懂的大模型知识分享，LLMs春/秋招大模型面试前必看，让你和面试官侃侃而谈。
    *   **定位**: **求职宝典**。在面试前刷一遍，能极大地提升你的信心和成功率。

*   **[LLM University](https://cohere.com/llmu)**
    *   **简介**: 一个由社区驱动的、非常全面的LLM学习大学。它系统地组织了从入门到高级的所有主题，并为每个主题提供了高质量的学习资源链接。
    *   **定位**: **宏观视野**。帮你跳出具体的技术点，从更宏观的视角理解AI的全景。
