# Agent 系列学习文章

面向 **AI Agent 应用架构师** 的系统性学习路线。从第一性原理出发、从实际问题出发，逐篇深入。共 10 模块、54 篇深度长文，覆盖从基础认知到生产工程、从单 Agent 到 Multi-Agent、从安全合规到前沿方向的全栈认知。

## 学习路线总览

```
模块一 ──► 模块二 ──► 模块三 ──► 模块四
(基础层)   (单Agent)  (Multi)   (Loop)
                 │
                 ├──► 模块五 (记忆与知识)
                 ├──► 模块六 (安全与治理)
                 ├──► 模块七 (评估与可观测)
                 └──► 模块八 (生产工程)
                           │
                           ▼
                      模块九 (交互设计)
                      模块十 (前沿方向)
```

## 文章目录

### 模块一：基础层——大模型能力边界与交互范式

| # | 文章标题 | 状态 |
|---|---|---|
| 1.1 | [LLM 推理机制与能力边界](./1.1-LLM推理机制与能力边界.md) | ✅ 已完成 |
| 1.2 | [Prompt Engineering 系统化](./1.2-Prompt-Engineering系统化.md) | ✅ 已完成 |
| 1.3 | [Context Engineering：从 RAG 到上下文窗口管理](./1.3-Context-Engineering从RAG到上下文窗口管理.md) | ✅ 已完成 |
| 1.4 | [Function Calling / Tool Use 协议全景](./1.4-Function-Calling与Tool-Use协议全景.md) | ✅ 已完成 |
| 1.5 | [推理模型时代的 Agent 范式变化](./1.5-推理模型范式.md) | ✅ 已完成 |
| 1.6 | [结构化输出工程](./1.6-结构化输出工程.md) | ✅ 已完成 |

**学完能回答**：模型能做什么、不能做什么？上下文怎么喂最高效？工具怎么暴露给模型？推理模型如何重塑 Agent 设计？

---

### 模块二：单 Agent 架构——从 ReAct 到 Harness Engineering

| # | 文章标题 | 状态 |
|---|---|---|
| 2.1 | [Agent 基础范式：ReAct / Plan-and-Execute / Reflexion / LATS](./2.1-Agent基础范式ReAct-Plan-Reflexion-LATS.md) | ✅ 已完成 |
| 2.2 | [Harness Engineering 与 Loop Engineering 完整体系](./2.2-Harness-Engineering与Loop-Engineering.md) | ✅ 已完成 |
| 2.3 | [Agent 隔离技术与递归自改进](./2.3-Agent隔离技术与递归自改进.md) | ✅ 已完成 |
| 2.4 | [MCP（Model Context Protocol）深度解析](./2.4-MCP深度解析.md) | ✅ 已完成 |
| 2.5 | [Tool Schema 设计原则与工具生态治理](./2.5-Tool-Schema设计原则与工具生态治理.md) | ✅ 已完成 |

**学完能回答**：如何让单个 Agent 可靠地完成任务？Harness 的 Guides/Sensors 怎么设计？工具层怎么规划？

---

### 模块三：Multi-Agent 系统——协作、编排与拓扑

| # | 文章标题 | 状态 |
|---|---|---|
| 3.1 | [Multi-Agent 设计模式：Orchestrator / Hierarchical / Swarm / Event-Driven](./3.1-Multi-Agent设计模式.md) | ✅ 已完成 |
| 3.2 | [Agent 通信与协调：共享状态 vs 消息传递 vs A2A 协议](./3.2-Agent通信与协调.md) | ✅ 已完成 |
| 3.3 | [框架深度对比：LangGraph / CrewAI / AutoGen / OpenAI Agents SDK](./3.3-框架深度对比.md) | ✅ 已完成 |
| 3.4 | [10 个 Design Axes 实战运用](./3.4-10个Design-Axes实战.md) | ✅ 已完成 |
| 3.5 | [Agent 角色与人格工程](./3.5-Agent角色与人格工程.md) | ✅ 已完成 |
| 3.6 | [任务分解与结果聚合](./3.6-任务分解与结果聚合.md) | ✅ 已完成 |
| 3.7 | [Multi-Agent 可靠性：Blast Radius / 级联失败 / 容错与系统级评估](./3.7-Multi-Agent可靠性.md) | ✅ 已完成 |

**学完能回答**：何时用 Multi-Agent？如何选择编排模式？不同框架适合什么场景？多 Agent 系统如何抗故障？

---

### 模块四：Loop Engineering——从被动响应到自治系统

| # | 文章标题 | 状态 |
|---|---|---|
| 4.1 | [Loop Engineering 完整体系（含 Ralph Loop、5+1 Building Blocks）](./2.2-Harness-Engineering与Loop-Engineering.md) | ✅ 已完成 |
| 4.2 | [Workspace 四原语与预算策略设计](./4.2-Workspace四原语与预算策略.md) | ✅ 已完成 |
| 4.3 | [工业级 Loop 实现对比：Codex /goal vs Claude Code /loop vs Devin](./4.3-工业级Loop实现对比.md) | ✅ 已完成 |
| 4.4 | [Loop 治理：Verification Burden / Comprehension Debt / Token Economics](./4.4-Loop治理.md) | ✅ 已完成 |

**学完能回答**：Loop 和 ReAct 的本质区别是什么？如何让 Agent 安全地持续自主运行？

---

### 模块五：记忆与知识系统

| # | 文章标题 | 状态 |
|---|---|---|
| 5.1 | [Agent 记忆体系分类：Working / Episodic / Semantic / Procedural](./5.1-Agent记忆体系分类.md) | ✅ 已完成 |
| 5.2 | [RAG 工程深水区：Chunking / Hybrid Search / Reranking / GraphRAG](./5.2-RAG工程深水区.md) | ✅ 已完成 |
| 5.3 | [长期记忆架构：MemGPT / Letta / Hierarchical Memory](./5.3-长期记忆架构.md) | ✅ 已完成 |

**学完能回答**：Agent 的记忆怎么设计？RAG 做到什么程度才算生产级？

---

### 模块六：安全、护栏与治理

| # | 文章标题 | 状态 |
|---|---|---|
| 6.1 | [Prompt Injection 攻防：直接注入 / 间接注入 / OWASP Top 10 for LLM](./6.1-Prompt-Injection攻防.md) | ✅ 已完成 |
| 6.2 | [Guardrails 体系：Input / Output / Tool 三层护栏](./6.2-Guardrails三层护栏.md) | ✅ 已完成 |
| 6.3 | [Agent 权限模型：最小权限 / Scoped Credentials / Permission Gate](./6.3-Agent权限模型.md) | ✅ 已完成 |
| 6.4 | [Red Teaming、合规审计与人机信任边界](./6.4-RedTeaming合规审计与人机信任边界.md) | ✅ 已完成 |
| 6.5 | [Agent Identity 与供应链安全：MCP/工具供应链审查 / Verifiable Credentials](./6.5-AgentIdentity与供应链安全.md) | ✅ 已完成 |
| 6.6 | [Memory & Data Security：Memory Poisoning / MINJA / Data Exfiltration / PII / GDPR 实操](./6.6-Memory与DataSecurity.md) | ✅ 已完成 |
| 6.7 | [Kill Switch 与可控性工程：Interrupt / Pause / Quarantine / Rollback 的架构层实现](./6.7-KillSwitch与可控性工程.md) | ✅ 已完成 |
| 6.8 | [AI Safety 治理框架对照：NIST AI RMF / ISO 42001 / EU AI Act / 实操 Checklist](./6.8-AISafety治理框架对照.md) | ✅ 已完成 |

**学完能回答**：如何确保 Agent 不做坏事？权限怎么设计？出了问题怎么归因？合规怎么落？

---

### 模块七：评估、测试与可观测性

| # | 文章标题 | 状态 |
|---|---|---|
| 7.1 | [Agent 评估框架：Task Completion / LLM-as-Judge / Benchmark 设计](./7.1-Agent评估框架.md) | ✅ 已完成 |
| 7.2 | [Agent 测试工程：Unit / Integration / Regression / Mutation](./7.2-Agent测试工程.md) | ✅ 已完成 |
| 7.3 | [可观测性平台：Trace / Metrics / LangSmith / Arize / Braintrust](./7.3-可观测性平台.md) | ✅ 已完成 |
| 7.4 | [持续改进闭环：A/B Test / Online Evaluation / Drift Detection](./7.4-持续改进闭环.md) | ✅ 已完成 |
| 7.5 | [领域专项 Benchmark 深度解读：SWE-bench Verified/Live / τ-bench / GAIA / WebArena / OSWorld / MLE-Bench / HLE](./7.5-领域专项Benchmark深度解读.md) | ✅ 已完成 |

**学完能回答**：怎么知道 Agent 有没有变好或变差？如何量化 Agent 表现？该用哪个 benchmark？

---

### 模块八：生产工程——部署、扩缩、成本

| # | 文章标题 | 状态 |
|---|---|---|
| 8.1 | [Agent 部署架构：Stateless + External State / Queue / Serverless vs Long-running](./8.1-Agent部署架构.md) | ✅ 已完成 |
| 8.2 | [可靠性设计：Retry / Circuit Breaker / Idempotency / Graceful Degradation](./8.2-可靠性设计.md) | ✅ 已完成 |
| 8.3 | [成本工程：Prompt Caching / Batch API / 级联路由 / 预算管理](./8.3-成本工程.md) | ✅ 已完成 |
| 8.4 | [Latency Engineering：TTFT vs E2E / Streaming / Speculative Decoding / Parallel Tool Calls / Prompt Cache 命中工程](./8.4-Latency-Engineering.md) | ✅ 已完成 |
| 8.5 | [模型选型与混合策略：Self-hosted（vLLM/SGLang）vs Managed / 开源 vs 闭源 / 级联路由 / 数据主权](./8.5-模型选型与混合策略.md) | ✅ 已完成 |
| 8.6 | [数据飞轮：Logs → Auto-labeling → Eval 集 → SFT/DPO/RFT → 持续改进闭环](./8.6-数据飞轮.md) | ✅ 已完成 |

**学完能回答**：如何把 demo 变成 production-grade 系统？怎么控制成本？怎么压延迟？怎么把生产数据变成训练数据？

---

### 模块九：Human-Agent 交互设计

| # | 文章标题 | 状态 |
|---|---|---|
| 9.1 | [Agent UX 原则：透明性 / 可预期性 / 可控性](./9.1-Agent-UX原则.md) | ✅ 已完成 |
| 9.2 | [自主度梯度：Copilot → Auto-pilot → Full Autonomous](./9.2-自主度梯度.md) | ✅ 已完成 |
| 9.3 | [中断、暂停、接管、回滚的 UX 模式](./9.3-中断暂停接管回滚.md) | ✅ 已完成 |
| 9.4 | [Agent 可解释性与决策溯源 UX：Thinking 流式呈现 / Trace 可视化 / Citation 链 / 错误归因](./9.4-可解释性UX.md) | ✅ 已完成 |
| 9.5 | [异步与后台 Agent UX：Devin/Cursor Background 范式 / 通知中断 / 多任务面板 / 进度可视化](./9.5-异步后台Agent-UX.md) | ✅ 已完成 |

**学完能回答**：Agent 的能力应该以什么方式呈现给用户？何时放权、何时收紧？后台 Agent 怎么设计才让人安心？

---

### 模块十：前沿方向

| # | 文章标题 | 状态 |
|---|---|---|
| 10.1 | [递归自改进（Recursive Self-Improvement）](./2.3-Agent隔离技术与递归自改进.md) | ✅ 已完成 |
| 10.2 | [World Models 在 Agent 中的角色](./10.2-World-Models.md) | ✅ 已完成 |
| 10.3 | [Agent-to-Agent 经济与 A2A 协议](./10.3-A2A协议与Agent经济.md) | ✅ 已完成 |
| 10.4 | [OS-level Agent / Computer Use / Desktop Agent](./10.4-OS-level-Agent.md) | ✅ 已完成 |
| 10.5 | [Agent 标准与治理演进（MCP / EU AI Act / Agent Identity）](./10.5-Agent标准与治理.md) | ✅ 已完成 |

---

## 核心参考资源

| 来源 | 链接 | 说明 |
|---|---|---|
| Anthropic | [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) | Agent 构建哲学 |
| Martin Fowler | [Harness Engineering](https://martinfowler.com/articles/harness-engineering.html) | Harness 工程体系 |
| OpenAI | [Harness Engineering](https://openai.com/index/harness-engineering/) | Codex Loop 设计 |
| LangChain | [State of Agent Engineering](https://www.langchain.com/state-of-agent-engineering) | 行业报告 |
| Azure | [AI Agent Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns) | 编排模式 |
| MCP | [Specification](https://modelcontextprotocol.io/specification/2025-03-26) | 工具协议规范 |
| boydfd | [拆解 Harness & Loop](https://www.cnblogs.com/boydfd/p/20525224) | 中文深度解读 |
| puppyone | [Loop Engineering](https://www.puppyone.ai/en/blog/what-is-loop-engineering-5-building-blocks-missing-one) | Loop 构件 |
| Lilian Weng | [LLM Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | 经典综述 |
| OWASP | [Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | 安全参考 |

## 学习节奏建议

| 阶段 | 模块 | 产出形式 | 预计周期 |
|---|---|---|---|
| 第一阶段（夯基） | 模块一 + 模块二 | 每主题 1 篇深度文章 | 2-3 周 |
| 第二阶段（展开） | 模块三 + 模块四 + 模块五 | 深度文章 + 实践笔记 | 3-4 周 |
| 第三阶段（硬化） | 模块六 + 模块七 + 模块八 | 偏实操，含代码示例 | 3-4 周 |
| 第四阶段（闭环） | 模块九 + 模块十 | 视野拓展，持续跟踪 | 持续 |

---

*已完成的文章存放于语雀 [LLM应用](https://aliyuque.antfin.com/bedrut/ott72s) 知识库。*
