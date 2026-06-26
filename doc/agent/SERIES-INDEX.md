# SERIES-INDEX — Agent 系列总索引（54 篇）

> 这是一份**密度型**索引，比 README 更聚焦"想找某个具体问题的答案"。
> README 提供学习路线，本文档提供**速查地图**。

---

## 一、按问题找文章（FAQ → Article）

### A. 模型能力与认知边界

| 我想搞清楚… | 去看 |
|---|---|
| LLM 推理是怎么回事、能做什么不能做什么 | [1.1](./1.1-LLM推理机制与能力边界.md) |
| Prompt 怎么写才系统化、有方法论 | [1.2](./1.2-Prompt-Engineering系统化.md) |
| 上下文窗口怎么管理、context engineering 是啥 | [1.3](./1.3-Context-Engineering从RAG到上下文窗口管理.md) |
| Function Calling / Tool Use 各家协议有何差异 | [1.4](./1.4-Function-Calling与Tool-Use协议全景.md) |
| GPT-5 / DeepSeek R1 / Claude Extended Thinking 重塑了什么 | [1.5](./1.5-推理模型范式.md) |
| 怎么让模型稳定输出 JSON / 结构化数据 | [1.6](./1.6-结构化输出工程.md) |

### B. 单 Agent 工程

| 我想搞清楚… | 去看 |
|---|---|
| ReAct / Plan-and-Execute / Reflexion / LATS 区别 | [2.1](./2.1-Agent基础范式ReAct-Plan-Reflexion-LATS.md) |
| Harness / Loop 怎么端到端搭 | [2.2](./2.2-Harness-Engineering与Loop-Engineering.md) |
| Agent 怎么沙箱隔离、递归自改进 | [2.3](./2.3-Agent隔离技术与递归自改进.md) |
| MCP 协议怎么读、怎么用 | [2.4](./2.4-MCP深度解析.md) |
| 工具描述怎么设计、工具生态怎么治理 | [2.5](./2.5-Tool-Schema设计原则与工具生态治理.md) |

### C. Multi-Agent 协作

| 我想搞清楚… | 去看 |
|---|---|
| Orchestrator / Hierarchical / Swarm 怎么选 | [3.1](./3.1-Multi-Agent设计模式.md) |
| Agent 之间怎么通信（共享状态 vs 消息 vs A2A） | [3.2](./3.2-Agent通信与协调.md) |
| LangGraph / CrewAI / AutoGen / OpenAI Agents SDK 选哪个 | [3.3](./3.3-框架深度对比.md) |
| Multi-Agent 10 个 Design Axes 怎么用 | [3.4](./3.4-10个Design-Axes实战.md) |
| Persona / Role / Capability Boundary 怎么分清 | [3.5](./3.5-Agent角色与人格工程.md) |
| 任务怎么拆、结果怎么合 | [3.6](./3.6-任务分解与结果聚合.md) |
| Blast Radius / 级联失败 / 容错怎么搞 | [3.7](./3.7-Multi-Agent可靠性.md) |

### D. Loop Engineering

| 我想搞清楚… | 去看 |
|---|---|
| Loop 完整体系（含 Ralph Loop / 5+1 Building Blocks）| [4.1 = 2.2](./2.2-Harness-Engineering与Loop-Engineering.md) |
| Workspace 四原语 / 预算策略 | [4.2](./4.2-Workspace四原语与预算策略.md) |
| Codex /goal vs Claude /loop vs Devin 实现差异 | [4.3](./4.3-工业级Loop实现对比.md) |
| Verification Burden / Comprehension Debt / Token Economics | [4.4](./4.4-Loop治理.md) |

### E. 记忆与知识

| 我想搞清楚… | 去看 |
|---|---|
| Working / Episodic / Semantic / Procedural 记忆 | [5.1](./5.1-Agent记忆体系分类.md) |
| Chunking / Hybrid Search / Reranking / GraphRAG | [5.2](./5.2-RAG工程深水区.md) |
| MemGPT / Letta / Hierarchical Memory | [5.3](./5.3-长期记忆架构.md) |

### F. 安全、护栏与治理

| 我想搞清楚… | 去看 |
|---|---|
| Prompt Injection 直接 / 间接 / OWASP Top 10 LLM | [6.1](./6.1-Prompt-Injection攻防.md) |
| Input / Output / Tool 三层护栏 | [6.2](./6.2-Guardrails三层护栏.md) |
| Agent 权限模型、Scoped Credentials、Permission Gate | [6.3](./6.3-Agent权限模型.md) |
| Red Teaming / 合规审计 / 人机信任 | [6.4](./6.4-RedTeaming合规审计与人机信任边界.md) |
| Agent Identity / 工具供应链 / Verifiable Credentials | [6.5](./6.5-AgentIdentity与供应链安全.md) |
| Memory Poisoning / MINJA / Data Exfiltration / PII / GDPR | [6.6](./6.6-Memory与DataSecurity.md) |
| Interrupt / Pause / Quarantine / Rollback 架构 | [6.7](./6.7-KillSwitch与可控性工程.md) |
| NIST AI RMF / ISO 42001 / EU AI Act 对照 | [6.8](./6.8-AISafety治理框架对照.md) |

### G. 评估、测试与可观测

| 我想搞清楚… | 去看 |
|---|---|
| Task Completion / LLM-as-Judge / Benchmark 设计 | [7.1](./7.1-Agent评估框架.md) |
| Unit / Integration / Regression / Mutation 测试 | [7.2](./7.2-Agent测试工程.md) |
| LangSmith / Arize / Braintrust 可观测平台 | [7.3](./7.3-可观测性平台.md) |
| A/B Test / Online Eval / Drift Detection | [7.4](./7.4-持续改进闭环.md) |
| SWE-bench / τ-bench / GAIA / WebArena / OSWorld / MLE / HLE | [7.5](./7.5-领域专项Benchmark深度解读.md) |

### H. 生产工程

| 我想搞清楚… | 去看 |
|---|---|
| Stateless + External State / Queue / Serverless vs Long-running | [8.1](./8.1-Agent部署架构.md) |
| Retry / Circuit Breaker / Idempotency / Graceful Degradation | [8.2](./8.2-可靠性设计.md) |
| Prompt Caching / Batch API / 级联路由 / 预算管理 | [8.3](./8.3-成本工程.md) |
| TTFT / Streaming / Speculative Decoding / Parallel Tool Calls | [8.4](./8.4-Latency-Engineering.md) |
| Self-hosted vs Managed / 开源 vs 闭源 / 数据主权 | [8.5](./8.5-模型选型与混合策略.md) |
| Logs → Auto-labeling → SFT/DPO/RFT 数据飞轮 | [8.6](./8.6-数据飞轮.md) |

### I. 交互设计

| 我想搞清楚… | 去看 |
|---|---|
| 透明性 / 可预期性 / 可控性 三原则 | [9.1](./9.1-Agent-UX原则.md) |
| Copilot → Auto-pilot → Full Autonomous 自主度梯度 | [9.2](./9.2-自主度梯度.md) |
| 中断 / 暂停 / 接管 / 回滚 UX 模式 | [9.3](./9.3-中断暂停接管回滚.md) |
| Thinking 流式 / Trace 可视化 / Citation 链 | [9.4](./9.4-可解释性UX.md) |
| Devin / Cursor Background 异步范式 | [9.5](./9.5-异步后台Agent-UX.md) |

### J. 前沿方向

| 我想搞清楚… | 去看 |
|---|---|
| Recursive Self-Improvement | [10.1 = 2.3](./2.3-Agent隔离技术与递归自改进.md) |
| World Models（V-JEPA 2 / Genie 3 / Agentic World Modeling）| [10.2](./10.2-World-Models.md) |
| A2A 协议 / Agent 经济（AP2 / x402 / MPP）| [10.3](./10.3-A2A协议与Agent经济.md) |
| OS-level Agent / Computer Use / OSWorld | [10.4](./10.4-OS-level-Agent.md) |
| MCP 三次修订 / EU AI Act / Agent Identity | [10.5](./10.5-Agent标准与治理.md) |

---

## 二、按角色找学习路径

### 路径 1：从零到 Production Agent 工程师（推荐顺序）

```
1.1 → 1.2 → 1.4 → 2.1 → 2.2 → 2.5 → 6.1 → 6.2 → 7.1 → 8.1 → 8.2
```

12 篇打底，覆盖：模型 → Prompt → 工具 → ReAct → Loop → 工具治理 → 安全双层 → 评估 → 部署 → 可靠性。

### 路径 2：单 Agent → Multi-Agent 跃迁

```
2.2 → 2.3 → 3.1 → 3.2 → 3.5 → 3.6 → 3.7 → 5.1 → 5.3
```

9 篇，从单 Agent 内核扩展到协作系统，含记忆体系基础。

### 路径 3：Agent 应用架构师全景（本系列的核心目标）

```
模块一全部 (6) → 2.1,2.2,2.4,2.5 → 模块三全部 (7) → 4.x → 5.x → 6.1,6.2,6.3,6.7 → 7.1,7.5 → 8.1,8.2,8.5 → 9.1,9.2 → 10.3,10.5
```

约 32 篇，是 README 第二阶段 + 第三阶段 + 第四阶段精华。

### 路径 4：安全 / 治理 / 合规 专精

```
6.1 → 6.2 → 6.3 → 6.5 → 6.6 → 6.7 → 6.4 → 6.8 → 10.5
```

9 篇，OWASP Top 10 LLM + Identity + Memory Security + Kill Switch + EU AI Act 全栈。

### 路径 5：评估 / 数据飞轮 专精

```
7.1 → 7.2 → 7.3 → 7.4 → 7.5 → 8.6 → 9.4
```

7 篇，benchmark 选型 + 在线评估 + Drift Detection + 数据飞轮闭环。

---

## 三、按模型 / 系统 / 概念查找

| 关键词 | 出现于 |
|---|---|
| GRPO / DeepSeek R1 | 1.5 |
| Extended Thinking / clear_thinking | 1.5, 9.4 |
| reasoning_effort / verbosity | 1.5, 8.4 |
| Pydantic / JSONSchemaBench | 1.6 |
| ReAct / Reflexion / LATS | 2.1 |
| Harness / Loop / Ralph | 2.2, 4.3 |
| MCP / Specification | 2.4, 10.5 |
| Tool Naming / Description | 2.5 |
| Orchestrator-Worker | 3.1, 3.5, 3.6 |
| Swarm / Event-Driven | 3.1 |
| A2A | 3.2, 10.3 |
| LangGraph / CrewAI / AutoGen / OpenAI Agents SDK | 3.3 |
| 10 Design Axes | 3.4 |
| RoleSpec / Capability Boundary | 3.5 |
| Map-Reduce / Synthesizer / Debate | 3.6 |
| MAST / MAS-FIRE / Saga / Circuit Breaker | 3.7, 8.2 |
| Workspace 四原语 | 4.2 |
| Codex /goal / Claude /loop / Devin | 4.3 |
| Verification Burden / Comprehension Debt | 4.4 |
| MemGPT / Letta | 5.3 |
| GraphRAG / Hybrid Search | 5.2 |
| Prompt Injection / OWASP Top 10 LLM | 6.1 |
| Permission Gate / Scoped Credentials | 6.3, 6.5 |
| Memory Poisoning / MINJA | 6.6 |
| Kill Switch / Quarantine | 6.7 |
| NIST AI RMF / ISO 42001 / EU AI Act | 6.8, 10.5 |
| SWE-bench / τ-bench / GAIA / OSWorld | 7.5 |
| LangSmith / Arize / Braintrust | 7.3 |
| vLLM / SGLang | 8.5 |
| Prompt Caching / Batch API | 8.3 |
| Speculative Decoding / Parallel Tool Calls | 8.4 |
| Devin / Cursor Background | 9.5 |
| V-JEPA 2 / Genie 3 | 10.2 |
| AP2 / x402 / MPP | 10.3 |
| Agentic World Model L1/L2/L3 | 10.2 |
| W3C Agent Identity Registry / AgentDID | 10.5 |

---

## 四、统计

| 维度 | 数值 |
|---|---|
| 总篇数 | 54 |
| 总模块数 | 10 |
| 估计总字数 | ~250,000 中文字符（约 350 页书） |
| 引用来源数 | 250+（论文 / 文档 / Blog） |
| 代码骨架数 | 80+ |
| 决策矩阵 / 对比表 | 100+ |
| 反模式条目 | 250+ |

---

> 找完想要的章节？回 [README 学习路线总览](./README.md) 或读 [SERIES-RETROSPECTIVE.md 收官辞](./SERIES-RETROSPECTIVE.md)。
