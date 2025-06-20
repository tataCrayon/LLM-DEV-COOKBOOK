from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, LLMResult
from typing import Any, Dict, List


class IterationCounterCallback(BaseCallbackHandler):
    """一个在每次Agent行动前打印轮次的回调处理器"""
    def __init__(self):
        self.iteration_count = 0

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.iteration_count += 1
        print(f"\n--- 🤔 思考轮次: {self.iteration_count} ---")

class AgentTraceCallback(BaseCallbackHandler):
    """一个捕获并存储Agent与LLM之间完整交互记录的回调处理器"""
    def __init__(self):
        self.trace = ""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        self.trace += f"\n--- PROMPT TO LLM ---\n{prompts[0]}\n"

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        self.trace += f"--- RESPONSE FROM LLM ---\n{response.generations[0][0].text}\n"