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


- https://github.com/datawhalechina/so-large-lm  
旨在作为一个大规模预训练语言模型的教程，从数据准备、模型构建、训练策略到模型评估与改进，以及模型在安全、隐私、环境和法律道德方面的方面来提供开源知识。


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


LangChain团队的RAG系列视频：[RAG From Scratch](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)  

大佬“沧海九粟”的中文版：[从零开始学习 RAG](https://www.bilibili.com/video/BV1dm41127jc/)

### 深入学习  

- GraphRAG  
微软研究院的GraphRAG介绍：[《Welcome to GraphRAG》](https://microsoft.github.io/graphrag/)

- AgenticRAG
[AgenticRAG](https://blog.bytebytego.com/p/ep169-rag-vs-agentic-rag)

---

## 5️⃣Agent

这一块目前还是东看看西看看，如果有推荐欢迎联系补充。  
目前看的书《大模型应用开发 动手做AI Agent》。


### 扩展学习  

来自anthropic官网的文章[《building-effective-agents》](https://www.anthropic.com/engineering/building-effective-agents)。



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

*   **[谷歌机器学习教程)](https://developers.google.com/machine-learning?hl=zh-cn)**
    *   **简介**: 通过 Google 的机器学习课程学习如何构建机器学习产品。

*   **[LLM 知识汇总(LLMForEverybody)](https://github.com/luhengshiwo/LLMForEverybody)**
    *   **简介**: 每个人都能看懂的大模型知识分享，LLMs春/秋招大模型面试前必看，让你和面试官侃侃而谈。
    *   **定位**: **求职宝典**。在面试前刷一遍，能极大地提升你的信心和成功率。

*   **[LLM University](https://cohere.com/llmu)**
    *   **简介**: 一个由社区驱动的、非常全面的LLM学习大学。它系统地组织了从入门到高级的所有主题，并为每个主题提供了高质量的学习资源链接。
    *   **定位**: **宏观视野**。帮你跳出具体的技术点，从更宏观的视角理解AI的全景。
