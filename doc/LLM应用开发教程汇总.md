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

Prompt Engineering 已经从"调话术"演进为"上下文工程"（Context Engineering）——围绕模型构造可控、可测、可复用的输入接口，是 LLM 应用开发的第一性技能。

### 核心必读（按优先级排序）

- **[《提示工程指南》（Prompt Engineering Guide, dair-ai）](https://www.promptingguide.ai/zh)**（推荐首读）
  社区维护最全面的提示工程入门站，覆盖 zero-shot/few-shot/CoT/Self-Consistency/ReAct/Tree of Thoughts 等所有主流技巧，并跟进最新论文。中文版完整。

- **[Anthropic: Prompt Engineering Overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)**
  Anthropic 官方提示工程指南，从结构化提示、XML 标签、role/system message、prefill 到 chain-of-thought 都有可直接套用的模板。是写生产 Prompt 的官方手册。

- **[OpenAI Cookbook: Prompt Engineering Techniques](https://github.com/openai/openai-cookbook)**
  OpenAI 官方 cookbook，包含大量 production-grade prompt 示例与函数调用、结构化输出最佳实践。GitHub 镜像更新最快、最稳定。

- **[Lilian Weng: Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)**
  OpenAI 安全研究负责人 Lilian Weng 的系统综述，将提示工程拆解为 Instruction / Few-shot / CoT / Augmented Generation 等范式。和她的 Agent 综述一起读最佳。

- **[Learn Prompting](https://learnprompting.org/docs/introduction)**
  由学术界与社区联合维护的开源提示工程教科书，结构清晰，含交互式练习。也是 ChatGPT Prompt Engineering for Developers 课程的扩展资料。

### 系统化课程

- **[DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)**（吴恩达 + OpenAI）
  最经典的开发者向 Prompt 短课程，1.5 小时讲完核心模式。

- **[吴恩达《面向开发者的提示工程》中文版（datawhalechina/llm-cookbook）](https://github.com/datawhalechina/llm-cookbook)**
  吴恩达系列课程的中文翻译 + 复现 + 调优，覆盖 Prompt → RAG → Fine-tuning 全流程。

- **[Anthropic: Prompt Engineering Interactive Tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial)**
  Anthropic 官方交互式教程，9 个 Jupyter Notebook 章节循序渐进，从基础到 Tool Use。

- **[Anthropic: Courses](https://github.com/anthropics/courses)**
  Anthropic 官方完整课程库，含 Prompt Engineering / Real-world Prompting / Tool Use / Prompt Evaluations 四门课。

- **[NirDiamant: Prompt Engineering Techniques](https://github.com/NirDiamant/Prompt_Engineering)**
  社区驱动的提示工程模式库，含 22+ 种技术 Notebook 实现（CoT/ToT/Self-Consistency/Constitutional AI 等）。

### 进阶范式

- **[Prompt Engineering: A Systematic Survey (arXiv:2402.07927)](https://arxiv.org/abs/2402.07927)**
  41 页系统综述，把提示工程组织为按任务/按模型/按方法的三维分类法。读完一篇胜过看十篇博客。

- **[Chain-of-Thought Prompting Elicits Reasoning (arXiv:2201.11903)](https://arxiv.org/abs/2201.11903)**
  CoT 原始论文。理解推理类 prompt 的起点。

- **[Self-Consistency Improves CoT (arXiv:2203.11171)](https://arxiv.org/abs/2203.11171)** / **[Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)** / **[ReAct (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)**
  CoT 之后的三大进化方向：投票、搜索、行动。

- **[LangChain: Communication is All You Need](https://blog.langchain.com/communication-is-all-you-need/)**
  把 Prompt 视作"人机沟通协议"的视角，对设计长期 Agent 系统的提示有启发。

- **[Karpathy on Context Engineering](https://x.com/karpathy/status/1937902205765607626)**
  Andrej Karpathy 提出 "Context Engineering > Prompt Engineering" 的 reframe，影响了 2026 Agent 设计范式。

### 安全与对抗

- **[Prompt Injection: OWASP LLM01](https://owasp.org/www-project-top-10-for-large-language-model-applications/)**
  OWASP LLM 应用 Top 10 第一名仍是 Prompt Injection；任何生产 Agent 都需了解。

- **[Prompt 越狱攻击技术手册](https://github.com/Acmesec/PromptJailbreakManual)**
  中文社区维护的越狱案例库，攻防视角同时学习。

- **[Anthropic: Red Teaming Language Models](https://docs.anthropic.com/en/docs/test-and-evaluate/strengthen-guardrails/reduce-jailbreak)**
  Anthropic 官方对抗性测试与护栏强化指南。

### 速成与速查

- **[OpenAI Platform: Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)**（官方简版，约 30 分钟读完）
- **[Anthropic: Prompt Library](https://docs.anthropic.com/en/resources/prompt-library/library)**：60+ 任务的官方 Prompt 模板。
- **[awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts)**：155K+ stars，最大社区 Prompt 集合。

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

LLM 应用框架已经从"百花齐放"演进到分层清晰：上层 Agent 编排、中层数据与结构化、下层推理服务、横向评估与可观测。选型不是"哪个最好"，而是"哪个最适合你当下的瓶颈"。

### 通用编排框架

- **[LangChain](https://python.langchain.com/docs/introduction/)**
  LLM 应用开发事实标准之一，组件最全（Models / Prompts / Memory / Retrievers / Chains / Tools）。**生态最广**、Agent 能力最成熟，缺点是抽象层多、版本演进快。

- **[LangGraph](https://langchain-ai.github.io/langgraph/)**
  LangChain 团队的 Agent 编排引擎，基于状态图 + checkpointing + human-in-the-loop。是构建长期运行、可中断、可恢复 Agent 的首选。

- **[LlamaIndex](https://docs.llamaindex.ai/en/stable/)**
  以数据为中心的 RAG/Agent 框架，**RAG 能力最深**、连接器最多（300+）、Workflow 引擎与 LangGraph 并列。

- **[Haystack](https://docs.haystack.deepset.ai/docs/intro)**
  deepset 出品的生产级 LLM 框架，组件化设计，企业搜索/RAG 场景成熟，对接 Elastic/Weaviate 良好。

### Agent 编排框架

- **[CrewAI](https://github.com/crewAIInc/crewAI)**
  最易上手的 Multi-Agent 框架，"角色 + 任务 + 工具"模型直观，社区活跃，企业版在售。

- **[AutoGen / AG2](https://github.com/microsoft/autogen)**
  微软的多 Agent 对话框架；2025 后续重构为 AG2 项目，强调 event-driven、async-first。研究界引用最多。

- **[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)**
  OpenAI 官方 Agent 框架，融合了原 Swarm 实验，原生支持 Handoffs、Guardrails、Tracing。轻量但表达力足够。

- **[OpenAI Swarm（实验性，已并入 Agents SDK）](https://github.com/openai/swarm)**
  极简的 multi-agent handoff 范式参考实现，适合阅读源码理解 Agent 协作的最小骨架。

- **[Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)**
  微软主力 Agent SDK，支持 C#/Python/Java，企业集成（Azure、M365）最强。Process Framework 对应于 LangGraph。

- **[Google ADK (Agent Development Kit)](https://github.com/google/adk-python)**
  Google 2025 发布的官方 Agent SDK，原生支持 A2A 协议，目标对标 LangGraph + Semantic Kernel。

- **[Pydantic AI](https://ai.pydantic.dev/)**
  把 Pydantic 的"类型即契约"思想用到 Agent 上，type-safe Agent、结构化输出与 dependency injection 一体化。FastAPI 团队同源风格。

- **[Agno（原 Phidata）](https://github.com/agno-agi/agno)**
  轻量多模态 Agent 框架，强调"几行代码起 Agent"，built-in memory/knowledge/tools，适合快速原型。

### 低代码 / Workflow 平台

- **[Dify](https://github.com/langgenius/dify)**
  开源 LLMOps + Agent + RAG 一站式平台，UI 编排、API 网关、知识库、可观测内置，国内外社区都极活跃（100K+ stars）。

- **[n8n](https://github.com/n8n-io/n8n)**
  通用工作流自动化平台，原生 AI nodes + LangChain 集成，是"把 LLM 接入企业流程"的最务实选择之一。

- **[FastGPT](https://github.com/labring/FastGPT)**
  基于 LLM 的知识库问答系统 + 可视化工作流，中文场景部署友好。

- **[Flowise](https://github.com/FlowiseAI/Flowise)**
  LangChain.js 的可视化拖拽编排，前端开发者友好。

### 数据 / 结构化输出

- **[Instructor](https://github.com/567-labs/instructor)**
  Pydantic + LLM 的结构化输出库，3M+ 月下载量，是 OpenAI/Anthropic/Gemini 结构化输出的事实标准社区方案。

- **[Outlines](https://github.com/dottxt-ai/outlines)**
  guided generation 框架，支持正则、CFG、JSON Schema 约束生成，vLLM/SGLang 内置后端。

- **[BAML](https://github.com/BoundaryML/baml)**
  把 prompt + schema + 测试当作一种独立 DSL 编译；解决 CoT + JSON 混合输出的鲁棒解析问题。

### 推理服务

- **[vLLM](https://github.com/vllm-project/vllm)**
  最广泛使用的开源 LLM 推理引擎，PagedAttention/连续 batching/Prefix caching 都源自此项目。生产部署首选。

- **[SGLang](https://github.com/sgl-project/sglang)**
  RadixAttention + 结构化输出加速的推理引擎，DeepSeek 等大模型官方部署后端之一。

- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)**
  NVIDIA 官方推理优化库，H100/H200/B200 极致性能。

- **[Ollama](https://github.com/ollama/ollama)**
  本地运行开源 LLM 的最简方式，一行命令拉模型。开发期 PoC 神器。

- **[LiteLLM](https://github.com/BerriAI/litellm)**
  100+ 模型厂商统一 API 网关，proxy 模式可做配额/缓存/路由，是多模型策略的胶水层。

### 评估与可观测

- **[LangSmith](https://docs.smith.langchain.com/)**：LangChain 官方追踪与评估平台，最成熟的商业方案。
- **[Langfuse](https://github.com/langfuse/langfuse)**：开源可自部署的 LLM Observability + Eval 平台。
- **[Arize Phoenix](https://github.com/Arize-ai/phoenix)**：开源 LLM/RAG/Agent 可观测，OpenTelemetry 原生。
- **[DeepEval](https://github.com/confident-ai/deepeval)**：pytest 风格的 LLM 评估框架，CI 友好。
- **[Ragas](https://github.com/explodinggradients/ragas)**：RAG 专用评估框架。
- **[MLflow LLM](https://mlflow.org/docs/latest/llms/index.html)**：传统 ML 平台对 LLM 的扩展，企业内已有 MLflow 时优先。
- **[Helicone](https://github.com/Helicone/helicone)**：YC 系开源 LLM 可观测，proxy 模式接入零侵入。

### 选型决策建议

| 场景 | 首选 | 备选 |
|---|---|---|
| 通用 LLM 应用 / 快速原型 | LangChain / LlamaIndex | Haystack |
| 长期运行 Agent / Workflow | LangGraph | Semantic Kernel / Pydantic AI |
| 多 Agent 协作 | CrewAI / AutoGen | OpenAI Agents SDK |
| RAG 深度优化 | LlamaIndex | LangChain + 自研检索层 |
| 企业低代码部署 | Dify / FastGPT | n8n |
| 结构化输出 | Instructor / Outlines | BAML |
| 推理服务（自托管） | vLLM / SGLang | TensorRT-LLM |
| 多厂商网关 | LiteLLM | OpenRouter |
| 评估与可观测 | LangSmith / Langfuse | Phoenix / DeepEval |

> 选型反模式：① 还没有需求就先选 Agent 框架（先用 LangChain 的 LCEL + 几个工具，瓶颈出现再升级）；② 选了框架就被框架绑死（保留模型层、检索层、Agent 层之间的接口隔离）；③ 把可观测留到出问题时再加（应当从 Day 1 开 trace）。

---

## 7️⃣项目

学完理论之后，最快速度形成工程直觉的方式是**读优秀开源项目源码**。下面按类型整理，每个项目都标注了它最值得学习的工程要点。

### 全景与集合

- **[awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)**
  精选 RAG / Agent / Multi-Agent / MCP / 语音 Agent 各类示例项目集合，是按"我想做 X，有没有参考实现"的方式快速找项目的入口。

- **[Awesome-LLM-Apps-Examples](https://github.com/sourabh-joshi/Generative-AI-Indepth-Basic-to-Advance)**
  另一份持续更新的 LLM 应用代码示例库，从基础到高级。

### Agent 编程助手 / Coding Agent（最值得读源码的一类）

- **[OpenHands（原 OpenDevin）](https://github.com/OpenHands/OpenHands)**
  60K+ stars 的开源 AI 软件工程师，可读其 Action/Observation 循环、Runtime 沙箱、Microagent 设计。是研究"Agent 如何控制 IDE"最完整的开源实现。

- **[Cline](https://github.com/cline/cline)**
  VSCode 内的 AI Coding Agent 插件，源码学习"Plan → Act 双模式 + 用户审批"的 UX 落地。

- **[OpenCode](https://github.com/anomalyco/opencode)**
  Terminal 内的 Coding Agent，对标 Claude Code，源码是学习 TUI Agent 的好范例。

- **[OpenInterpreter](https://github.com/openinterpreter/openinterpreter)**
  自然语言驱动的本地代码执行 Agent，60K+ stars，理解"Agent + REPL 沙箱"的经典实现。

- **[Aider](https://github.com/Aider-AI/aider)**
  CLI 内的结对编程 Agent，专注 git-aware 编辑、repo map、commit-by-commit 协作。代码风格清晰，适合学习"Agent + 源码上下文"的工程范式。

### 通用 Agent / Multi-Agent 实战

- **[MetaGPT](https://github.com/geekan/MetaGPT)**
  把"软件公司"角色化为 Agent，输入一句话生成 PRD + 设计 + 代码。学习多 Agent 协作 + SOP 编排的代表项目。

- **[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)**
  最早期的自主 Agent 项目，源码已被多次重构，仍是理解"目标 → 任务拆解 → 持续循环"思路的入门。

- **[GPT-Researcher](https://github.com/assafelovic/gpt-researcher)**
  自主研究 Agent，调研 → 多源检索 → 长文产出。读源码学"Agent 写长报告"的工程模式。

### RAG 系统

- **[Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat)**
  现象级中文 RAG 项目，**国内 RAG 工程模板**。代码层面理解全流程：文档解析、分块、Embedding、检索、重排、生成。

- **[RAGFlow](https://github.com/infiniflow/ragflow)**
  深度文档理解驱动的 RAG 引擎，重点是表格、版式、多模态文档处理；适合学习"非纯文本 RAG"的工程难点。

- **[LightRAG](https://github.com/HKUDS/LightRAG)**
  HKUDS 开源的轻量级 Graph-aware RAG，读论文 + 代码理解 GraphRAG 范式的工程实现。

- **[GraphRAG](https://github.com/microsoft/graphrag)**
  微软官方 GraphRAG 参考实现，理解实体抽取 → 社区检测 → 多跳推理的完整管线。

- **[FastGPT](https://github.com/labring/FastGPT)**
  企业知识库 + 工作流的开源实现，前后端俱全，是"对客交付的 RAG"最务实的范例。

- **[DB-GPT](https://github.com/eosphoros-ai/DB-GPT)**
  围绕"Database + LLM"构建的多 Agent 平台，含 Text2SQL、数据 Agent、ChatExcel、ChatPPT 等子能力。

### Memory & 长期记忆

- **[Mem0](https://github.com/mem0ai/mem0)**
  当前最活跃的 Agent Memory 框架（30K+ stars），实现了 short-term/long-term/scoped memory + 检索 + 衰减。

- **[Letta（原 MemGPT）](https://github.com/letta-ai/letta)**
  Berkeley 出身的有状态 Agent 系统，把 OS 虚拟内存思想搬到 LLM，理解"Agent 自编辑记忆"的范式典范。

### Chat 应用 / 全栈范例

- **[Lobe Chat](https://github.com/lobehub/lobe-chat)**
  60K+ stars 的现代化 Chat UI，插件系统 + 多模态 + 多模型路由。**学习全栈 LLM 应用的最佳前端工程范例**。

- **[Open WebUI](https://github.com/open-webui/open-webui)**
  本地 LLM 部署最流行的 Web 前端（120K+ stars），对接 Ollama/OpenAI-compatible，企业内私有部署常用。

- **[LibreChat](https://github.com/danny-avila/LibreChat)**
  开源 ChatGPT 克隆，支持多模型、Code Interpreter、Agent，社区贡献质量高。

- **[NextChat（原 ChatGPT-Next-Web）](https://github.com/ChatGPTNextWeb/NextChat)**
  Next.js + Vercel 一键部署的轻量级 Chat，80K+ stars，前端友好。

### 微调与训练

- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**
  最易用的 LLM 微调框架，**消费级显卡训练自有模型的首选**，LoRA/QLoRA/DPO/RLHF 全支持。

- **[Unsloth](https://github.com/unslothai/unsloth)**
  2x 加速 + 50% 显存节省的 LoRA 微调库，Colab 友好。

- **[axolotl](https://github.com/axolotl-ai-cloud/axolotl)**
  生产级微调框架，YAML 配置驱动，多卡/多机训练成熟。

### 数据 / ETL

- **[MarkItDown](https://github.com/microsoft/markitdown)**
  微软开源的"万物 → Markdown"工具，PDF/Office/图片/音频统一转 Markdown，RAG ETL 必备。

- **[Unstructured](https://github.com/Unstructured-IO/unstructured)**
  企业级文档解析库，分块/版式识别/表格抽取，是 LlamaIndex 等框架的底层依赖。

- **[LLaMA-OCR](https://github.com/Nutlope/llama-ocr)** / **[Marker](https://github.com/datalab-to/marker)**
  专攻 PDF → Markdown 的高质量开源 OCR 管线。

> 阅读建议：先选 1-2 个**对应你目标领域**的项目精读（Coding Agent 选 OpenHands、RAG 选 Langchain-Chatchat、Memory 选 Letta），其它项目用作"我要做某个特性时去翻参考实现"。不要囫囵吞枣读所有项目。

---

## 8️⃣八股

LLM 面试已经从"会用 ChatGPT"升级到"懂原理 + 能做工程 + 能讲项目"。题库刷一遍只是入场券，**真正过线靠的是有一个能讲清楚 30 分钟的项目**。

### 高质量题库

- **[LLM 面试问题汇总 (wdndev/llm_interview_note)](https://github.com/wdndev/llm_interview_note)**
  社区活跃维护的中文 LLM 面试题库，含题目、参考答案、讨论。**核心刷题资料**。

- **[LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)**
  "每个人都能看懂的大模型知识"，春秋招前必看，让你和面试官侃侃而谈。题目偏概念解释。

- **[deep-learning-interviews](https://github.com/BoltzmannEntropy/interviews.ai)**
  覆盖深度学习经典 + LLM 的英文面试题，适合冲外企。

### 系统化教程（按学习路径）

- **[InternLM/Tutorial](https://github.com/InternLM/Tutorial)**
  上海 AI Lab 出品的书生·浦语大模型实战营教程，从基础到 RAG/Agent/微调全覆盖，含真实项目。**首推**。

- **[datawhalechina/self-llm](https://github.com/datawhalechina/self-llm)**
  开源大模型食用指南，针对国内用户的环境配置 + 部署 + 微调全流程实战教程，30K+ stars。

- **[datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)**
  从零开始的 LLM 原理与实践教程，**深入理解 LLM 核心原理 + 动手实现你的第一个大模型**。

- **[liguodongiot/llm-action](https://github.com/liguodongiot/llm-action)**
  LLM 实战教程库，覆盖部署、训练、推理优化、量化、对齐、应用等主题，国内工程师视角，13K+ stars。

### 工程实战必备

- **[mlabonne/llm-course](https://github.com/mlabonne/llm-course)**
  Maxime Labonne 的 LLM 全景课程（45K+ stars），含 LLM Fundamentals / LLM Scientist / LLM Engineer 三条路径 + Colab Notebook。**英文首推**。

- **[karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)** & **[karpathy/llm.c](https://github.com/karpathy/llm.c)**
  Karpathy 的从零实现 GPT 项目，理解 Transformer / 训练 / 优化的最佳读物。面试时讲透这两个项目能直接证明实力。

- **[rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)**
  《Build a Large Language Model (From Scratch)》配套代码，35K+ stars，章节式渐进实现 ChatGPT 类 LLM。

### 实战经验与方法论

- **[LLM Bootcamp (Full Stack Deep Learning)](https://fullstackdeeplearning.com/llm-bootcamp/)**
  伯克利全栈深度学习团队的 LLM 训练营录像 + 讲义，工程深度极高，必看。

- **[Eugene Yan's LLM Patterns](https://eugeneyan.com/writing/llm-patterns/)**
  AI 工程师 Eugene Yan 整理的 LLM 应用 7 大模式（Evals / RAG / Fine-tuning / Caching / Guardrails / Defensive UX / Collect User Feedback）。**生产工程视角**。

- **[Chip Huyen: Designing Machine Learning Systems](https://github.com/chiphuyen/machine-learning-systems-design)**
  机器学习系统设计经典材料，面试系统设计题的核心参考。

- **[Chip Huyen: AI Engineering（书 + 资源）](https://github.com/chiphuyen/aie-book)**
  2025 年最新出版的《AI Engineering》一书配套资源，覆盖 LLM 应用开发的工程视角全景。

### 求职策略

- **[拒绝 996 的 LLM 工程师面试经验贴](https://github.com/luhengshiwo/LLMForEverybody/tree/main/01-%E5%88%86%E4%BA%AB%E4%B8%93%E5%8C%BA)**
  LLMForEverybody 仓库下的"分享专区"集合，含真实面经。

- **面试前 4 周策略**：
  1. **第 1 周**：刷题库（wdndev + LLMForEverybody）通读两遍，标记不会的；
  2. **第 2 周**：动手做 1 个能讲 30 分钟的项目（建议 RAG 或 Agent，选项目章节里的一个魔改）；
  3. **第 3 周**：写项目复盘文档，准备 STAR 法叙事框架（业务背景 / 技术选型 / 关键决策 / 数据 / 反思）；
  4. **第 4 周**：模拟面试 + 系统设计题（参考 Chip Huyen 资料）。

### 扩展阅读

- **[LLMSurvey (RUCAIBox)](https://github.com/RUCAIBox/LLMSurvey)**：人大综述论文配套仓库，理论背景厚（维护节奏放缓，但内容仍权威）。
- **[Awesome-LLM (Hannibal046)](https://github.com/Hannibal046/Awesome-LLM)**：LLM 资源 awesome list，仓库经典但 2025 年后更新放缓，做历史回溯参考。

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
