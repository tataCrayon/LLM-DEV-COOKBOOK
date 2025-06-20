from .agent_creator import create_agent_executor
from .callbacks.agent_callbacks import IterationCounterCallback, AgentTraceCallback
from .evaluators.agent_evaluator import evaluate_agent_trace 

def run_research_agent(question: str):
    """
    针对一个研究问题，初始化并运行研究员Agent。
    """
    print(f"--- 收到研究任务 ---")
    print(f"问题: {question}")
    
    agent_executor = create_agent_executor()
    
    # 创建回调处理器
    iteration_counter = IterationCounterCallback()
    trace_collector = AgentTraceCallback()
    
    try:
        # 运行Agent
        response = agent_executor.invoke(
            {"input": question},
            config={"callbacks": [iteration_counter, trace_collector]}
        )
        
        final_answer = response['output']
        agent_trace = trace_collector.trace
        
        print("\n--- Agent执行完毕 ---")
        print("最终答案:")
        print(final_answer)
        print("\n==================================================")
        
        # 我们可以打印或返回轨迹用于后续评估
        print("--- Agent完整思考轨迹 (已被捕获) ---")
        print(agent_trace)
        print("\n==================================================")
        
        # 调用LLM评委进行评估
        evaluate_agent_trace(
            question=question,
            agent_trace=agent_trace,
            final_answer=final_answer
        )
        
    except Exception as e:
        print(f"\n--- Agent执行过程中发生错误 ---")
        print(e)

# 这段代码使得我们可以直接运行这个文件来进行测试
if __name__ == '__main__':
    # 加载环境变量
    from dotenv import load_dotenv
    import os
    
    # 我们需要向上两级找到.env文件
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)

    research_question = "对比Java和Python在构建LLM应用中的优劣势，并列出各自的主流框架。"
    run_research_agent(research_question)