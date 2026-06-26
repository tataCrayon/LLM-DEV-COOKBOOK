# REFERENCES — Agent 系列全引文清单

> 按模块整理，覆盖论文、官方文档、规范、博客、工具链。约 250+ 条。
> 标注 [年份]，方便读者评估时效性。多次出现的引用按"首次出现的模块"归类。

---

## 模块一：基础层

### 论文 / 研究报告

- **[2017]** Vaswani, A. et al. *Attention Is All You Need*. NeurIPS. [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **[2022]** Wei, J. et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS. [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
- **[2022]** Kojima, T. et al. *Large Language Models are Zero-Shot Reasoners*. NeurIPS. [arxiv.org/abs/2205.11916](https://arxiv.org/abs/2205.11916)
- **[2023]** Wang, X. et al. *Self-Consistency Improves Chain of Thought Reasoning*. ICLR. [arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
- **[2024]** Liu, N. F. et al. *Lost in the Middle: How Language Models Use Long Contexts*. TACL. [arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
- **[2025]** DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability via Reinforcement Learning*. arXiv:2501.12948. [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- **[2025]** Qwen Team. *QwQ-32B Technical Report*.
- **[2025]** Korthikanti, V. et al. *JSONSchemaBench*. arXiv:2501.10868. [arxiv.org/abs/2501.10868](https://arxiv.org/abs/2501.10868)
- **[2025]** Anthropic. *On the Biology of a Large Language Model*. Transformer Circuits.
- **[2025]** OpenAI. *GPT-5 System Card*. [openai.com/index/gpt-5-system-card](https://openai.com/index/gpt-5-system-card/)

### 官方文档

- OpenAI. *Function calling guide*. [platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- OpenAI. *Structured Outputs guide*. [platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs)
- Anthropic. *Extended Thinking*. [docs.anthropic.com/en/docs/build-with-claude/extended-thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- Anthropic. *Tool Use Documentation*.
- Google. *Gemini Structured Output*.

### 博客 / 工具链

- Anthropic. *Building Effective Agents* (Dec 2024). [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents)
- Lilian Weng. *LLM Powered Autonomous Agents* (2023). [lilianweng.github.io/posts/2023-06-23-agent](https://lilianweng.github.io/posts/2023-06-23-agent/)
- Pydantic AI Docs. [ai.pydantic.dev](https://ai.pydantic.dev)
- Instructor (Jason Liu). [github.com/jxnl/instructor](https://github.com/jxnl/instructor)
- Outlines. [github.com/dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
- Guidance. [github.com/guidance-ai/guidance](https://github.com/guidance-ai/guidance)
- xgrammar. [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
- Wilkins. *Reasoning Models in Production* (2026).

---

## 模块二：单 Agent 架构

### 论文

- **[2022]** Yao, S. et al. *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR. [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- **[2023]** Shinn, N. et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS. [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- **[2023]** Zhou, A. et al. *Language Agent Tree Search (LATS)*. ICML 2024. [arxiv.org/abs/2310.04406](https://arxiv.org/abs/2310.04406)
- **[2023]** Wang, L. et al. *Plan-and-Solve Prompting*. ACL. [arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091)
- **[2024]** Wu, Y. et al. *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*. arXiv:2308.08155.

### 官方与规范

- Anthropic. *Model Context Protocol Specification* (2025-03-26 → 2025-06 → 2025-11 → 2026-01 MCP Apps). [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)
- OpenAI. *Agents SDK Documentation*. [openai.github.io/openai-agents-python](https://openai.github.io/openai-agents-python/)
- OpenAI. *Harness Engineering* (Codex Loop). [openai.com/index/harness-engineering](https://openai.com/index/harness-engineering/)

### 博客

- Martin Fowler. *Harness Engineering*. [martinfowler.com/articles/harness-engineering.html](https://martinfowler.com/articles/harness-engineering.html)
- boydfd. *拆解 Harness & Loop*. [cnblogs.com/boydfd/p/20525224](https://www.cnblogs.com/boydfd/p/20525224)
- PuppyOne. *What is Loop Engineering: 5+1 Building Blocks*. [puppyone.ai/en/blog](https://www.puppyone.ai/en/blog/what-is-loop-engineering-5-building-blocks-missing-one)
- Latent Space. *The Harness Engineering Era* (2025).
- Devin / Cognition. *Devin Architecture Notes* (2024-2025).

---

## 模块三：Multi-Agent

### 论文

- **[2025]** Cemri, M. et al. *Why Do Multi-Agent LLM Systems Fail? (MAST)*. NeurIPS 2025 D&B Spotlight. arXiv:2503.13657. [arxiv.org/abs/2503.13657](https://arxiv.org/abs/2503.13657)
- **[2026]** Wang, X. et al. *MAS-FIRE: Benchmarking Fault Injection Robustness on Multi-Agent Systems*. arXiv:2602.19843. [arxiv.org/abs/2602.19843](https://arxiv.org/abs/2602.19843)
- **[2025]** Anthropic. *How we built our multi-agent research system*. [anthropic.com/engineering/multi-agent-research-system](https://www.anthropic.com/engineering/multi-agent-research-system)
- **[2025]** Cognition. *Don't Build Multi-Agents*. [cognition.ai/blog/dont-build-multi-agents](https://cognition.ai/blog/dont-build-multi-agents)
- **[1982]** Lamport, L. et al. *The Byzantine Generals Problem*. ACM TOPLAS.
- **[1985]** Avizienis, A. *N-Version Approach to Fault-Tolerant Software*. IEEE TSE.
- **[1987]** Garcia-Molina, H. & Salem, K. *Sagas*. SIGMOD.
- **[2025]** WideSearch / A-MapReduce benchmark papers.

### 官方与规范

- A2A Working Group. *Agent-to-Agent Protocol v1.0* (Linux Foundation, 2025-2026).
- Azure Architecture Center. *AI Agent Design Patterns*. [learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)
- LangGraph Docs. [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- CrewAI Docs. [docs.crewai.com](https://docs.crewai.com)
- Microsoft AutoGen Docs. [microsoft.github.io/autogen](https://microsoft.github.io/autogen/)

### 博客

- Augment Code. *MAST: A Multi-Agent System Failure Taxonomy* (2026).
- Martin Fowler. *Circuit Breaker Pattern*. [martinfowler.com/bliki/CircuitBreaker.html](https://martinfowler.com/bliki/CircuitBreaker.html)
- Netflix Hystrix wiki.
- Zylos AI. *Graceful Degradation Patterns for Production Agent Systems* (Feb 2026).
- LangChain. *State of Agent Engineering* (2025). [langchain.com/state-of-agent-engineering](https://www.langchain.com/state-of-agent-engineering)

---

## 模块四：Loop Engineering

- (引用与模块二重叠，主要复用 Harness 系列)
- Codex Open Source. *Loop Design Notes*.
- Claude Code. */loop spec*.
- Devin. *Background Agent Architecture*.

---

## 模块五：记忆与知识

### 论文

- **[2023]** Packer, C. et al. *MemGPT: Towards LLMs as Operating Systems*. arXiv:2310.08560.
- **[2024]** Letta. *Letta Architecture* (MemGPT 升级版).
- **[2024]** GraphRAG: Edge, D. et al. *From Local to Global: A Graph RAG Approach*. Microsoft Research.
- **[2024]** Hybrid Search: Reciprocal Rank Fusion variants.
- **[2024]** Bge / Cohere Rerank model cards.

### 工具与文档

- LlamaIndex Docs.
- LangChain RAG Cookbook.
- Pinecone / Weaviate / Qdrant 向量库文档.
- Anthropic. *Contextual Retrieval* (2024).

---

## 模块六：安全、护栏与治理

### 标准与官方

- OWASP. *Top 10 for LLM Applications* (2025). [owasp.org/www-project-top-10-for-large-language-model-applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- NIST. *AI Risk Management Framework* (AI RMF 1.0 + GenAI Profile). [nist.gov/itl/ai-risk-management-framework](https://www.nist.gov/itl/ai-risk-management-framework)
- ISO/IEC. *42001:2023 AI Management Systems*.
- EU. *AI Act (Regulation 2024/1689)*. [eur-lex.europa.eu](https://eur-lex.europa.eu)
- EU AI Office. *GPAI Code of Practice* (2025).
- W3C. *Agent Identity Registry CG Charter* (2026-04).
- US Executive Order 14179 / AI Action Plan (2025).

### 论文

- **[2023]** Greshake, K. et al. *Not what you've signed up for: Indirect Prompt Injection*. arXiv:2302.12173.
- **[2024]** Perez, F. & Ribeiro, I. *Ignore Previous Prompt*. arXiv:2211.09527.
- **[2025]** MINJA. *Memory Injection Attacks on LLM Agents*.
- **[2025]** AgentDID. arXiv:2604.25189.
- **[2025]** Constitutional AI. Anthropic.

### 工具

- Lakera Guard / NeMo Guardrails / Llama Guard / Promptfoo / Garak (LLM Red Teaming).

---

## 模块七：评估、测试与可观测

### Benchmark 论文

- **[2024]** SWE-bench: Jimenez, C. et al. ICLR. [swebench.com](https://www.swebench.com/)
- **[2025]** SWE-bench Verified / Live.
- **[2024]** τ-bench: Yao, S. et al. Sierra. [sierra.ai/blog/benchmarking-ai-agents](https://sierra.ai/blog/benchmarking-ai-agents)
- **[2024]** GAIA: Mialon, G. et al. Meta FAIR.
- **[2024]** WebArena: Zhou, S. et al. CMU.
- **[2024]** OSWorld: Xie, T. et al.
- **[2024]** MLE-Bench: OpenAI.
- **[2025]** HLE (Humanity's Last Exam): Scale AI / CAIS.

### 工具

- LangSmith / Arize Phoenix / Braintrust / OpenLLMetry / Weave (Weights & Biases).
- Evals: Promptfoo / Inspect AI (UK AISI) / DeepEval.

---

## 模块八：生产工程

### 论文 / 工程

- **[2023]** Kwon, W. et al. *vLLM: PagedAttention*. SOSP. [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **[2024]** SGLang. *Structured Generation Language*. [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- **[2024]** Leviathan, Y. et al. *Speculative Decoding*. ICML.
- **[2025]** Anthropic. *Prompt Caching*.
- **[2025]** OpenAI. *Batch API*.

### 工程模式参考

- Google SRE Book. *Reliability Engineering*.
- Netflix Tech Blog. *Chaos Engineering*.
- AWS Well-Architected Framework.
- Stripe / Shopify webhook idempotency 实战.

---

## 模块九：UX

### 论文 / 报告

- **[2024]** GitHub. *Copilot Workspace Design Notes*.
- **[2025]** Devin / Cursor Background UX 公开案例.
- **[2025]** Anthropic. *Streaming Thinking* (UX 部分).
- Nielsen Norman Group. *AI Agent UX Principles*.

### 工具

- Vercel AI SDK UI Streaming.
- LangGraph Studio.

---

## 模块十：前沿方向

### World Models

- **[2025]** Meta AI. *V-JEPA 2: Self-Supervised Video World Model*. [ai.meta.com/research/publications/vjepa-2](https://ai.meta.com/research/publications/v-jepa-2/)
- **[2025]** Google DeepMind. *Genie 3: Real-Time Promptable World Models*.
- **[2025]** Agentic World Modeling Survey. arXiv.

### Agent Economics

- **[2025]** A2A Working Group. *A2A v1.0 Specification* (Linux Foundation).
- **[2025]** Google. *AP2 - Agent Payment Protocol*.
- **[2025]** Coinbase / x402 (HTTP 402 micropayment for agents).
- **[2025]** ACP (Agent Communication Protocol).
- **[2025]** MPP (Multi-Party Protocol).

### OS-level Agent

- **[2024]** Anthropic. *Computer Use* announcement.
- **[2025]** Apple. *App Intents for AI*.
- **[2024-2025]** OSWorld benchmark.
- **[2025]** Microsoft. *OmniParser*.

### 治理演进

- **[2025-2026]** MCP 三次修订（2025-03 OAuth 2.1 → 2025-06 Resource Indicator → 2025-11 CIMD + Tasks + Elicitation + Server-side loops → 2026-01 MCP Apps）.
- **[2025]** A2A v1.0 + LF 治理.
- **[2026]** W3C Agent Identity Registry CG.
- **[2026]** AgentDID arXiv:2604.25189.
- EU AI Act 关键时间线：2025-08 GPAI 生效 / 2026-08 执法 / 2027-08 存量大限.

---

## 跨模块的奠基级著作

- Lampson, B. *Hints for Computer System Design* (1983).
- Lamport, L. *Time, Clocks, and the Ordering of Events* (1978).
- Brewer, E. *CAP Theorem* (2000).
- Kleppmann, M. *Designing Data-Intensive Applications* (O'Reilly, 2017).
- Beyer, B. et al. *Site Reliability Engineering* (Google / O'Reilly, 2016).
- Hofmann, M. *Patterns of Distributed Systems* (martinfowler.com).
- Norvig, P. & Russell, S. *Artificial Intelligence: A Modern Approach* (4th ed.).

---

## 中文社区资源

- boydfd. cnblogs 系列博客.
- qaskills（注：企业代理偶尔拦截，回退用 arXiv 一手）.
- 阿里云 / 字节 / 美团 等 LLM 工程团队公开 case study.
- 公众号"DeepInsight"、"AI Agent 周报"等定期摘要.

---

## 工具与代码仓库（不完整，仅核心）

| 类别 | 仓库 |
|---|---|
| Agent 框架 | LangGraph / CrewAI / AutoGen / OpenAI Agents SDK / LlamaIndex Agents |
| 结构化输出 | Pydantic / Instructor / Outlines / Guidance / xgrammar |
| RAG | LlamaIndex / LangChain / Haystack |
| 向量库 | Pinecone / Weaviate / Qdrant / pgvector / Chroma |
| Memory | MemGPT / Letta / Zep |
| 评估 | Promptfoo / Inspect AI / DeepEval / Ragas |
| 可观测 | LangSmith / Arize Phoenix / Braintrust / OpenLLMetry |
| 安全 | Lakera Guard / NeMo Guardrails / Llama Guard / Garak |
| 模型服务 | vLLM / SGLang / Ollama / TGI / Llama.cpp |
| MCP / A2A | MCP SDK (Anthropic) / A2A SDK (Linux Foundation) |

---

## 引用使用约定

- 每条引用都附年份；若是 arXiv，附论文编号；若是博客，附 URL。
- 多次出现的引用只在首次出现的模块下完整列出，后续模块如重复使用，正文里直接简称。
- 本清单不构成"必读"列表——读完每篇文章的 §8 参考资料对应即可。

---

> 引用清单到此为止。回 [SERIES-INDEX](./SERIES-INDEX.md) 速查、回 [SERIES-RETROSPECTIVE](./SERIES-RETROSPECTIVE.md) 看收官辞、或回 [README](./README.md)。
