import os
from dotenv import load_dotenv

# --- 基础组件 ---
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document # 主要用于类型提示和理解

# --- 嵌入和向量存储 ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 加载环境变量
# 这会从 .env 文件中加载 DEEPSEEK_API_KEY
load_dotenv()

# --- LLM 初始化 ---
def get_deepseek_key():
    key = os.getenv('DEEPSEEK_API_KEY')
    if key is None or not key.startswith('sk-'): # 简单校验
        raise ValueError(
            "DEEPSEEK_API_KEY 未在环境变量中找到或格式不正确。请确保它是有效的 sk- 开头的密钥。")
    return key

def create_deepseek_llm():
    api_key = get_deepseek_key()
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0.1, # RAG 应用中通常希望答案更具确定性
        max_tokens=1024,
        api_key=api_key
    )

# --- 嵌入模型初始化 (与你之前的代码相同) ---
def initialize_embedding_model():
    print("\n--- 初始化嵌入模型 ---")
    model_name = "BAAI/bge-small-zh-v1.5" # 确保这个模型能被下载和加载
    model_kwargs = {'device': 'cuda'} # 改为 'cpu' 以增加通用性，如果 GPU 可用且配置好，可以改回 'cuda'
    encode_kwargs = {'normalize_embeddings': True}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"HuggingFace 嵌入模型 '{model_name}' 初始化成功。")
        return embeddings
    except Exception as e:
        print(f"初始化嵌入模型时出错: {e}")
        print("请确保 sentence-transformers 和 PyTorch (CPU 或 GPU 版本) 已正确安装。")
        return None

# --- 加载 Chroma 向量数据库 (与你之前的代码相同) ---
def load_chroma_vector_store(embedding_model, persist_directory="./my_chroma_data"):
    if not embedding_model:
        print("嵌入模型未初始化，无法加载 Chroma 数据库。")
        return None
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        print(f"持久化目录 '{persist_directory}' 不存在或为空。请先运行创建数据库的脚本。")
        return None

    print(f"\n--- 从 '{persist_directory}' 加载 Chroma 数据库 ---")
    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name="my_knowledge_base" # 确保与创建时一致
        )
        print("Chroma 数据库加载成功。")
        return vector_store
    except Exception as e:
        print(f"加载 Chroma 数据库时出错: {e}")
        return None

# --- 辅助函数：格式化检索到的文档 ---
def format_docs(docs: list[Document]) -> str:
    """将检索到的文档列表格式化为单一字符串上下文。"""
    if not docs:
        return "没有找到相关信息。"
    return "\n\n".join(f"来源 {i+1} (ID: {doc.metadata.get('doc_id', '未知')}):\n{doc.page_content}"
                       for i, doc in enumerate(docs))

# --- 构建和测试 RAG 链 ---
def test_rag_chain(llm_instance, retriever_instance):
    print("\n--- 测试 RAG 链 ---")

    # 1. 定义 RAG 提示模板
    rag_template = """
    你是一个问答助手。请根据下面提供的“已知信息”来回答用户提出的“问题”。
    你需要：
    1. 仔细阅读“已知信息”，只依据这些信息回答问题。
    2. 如果“已知信息”中没有相关内容来回答问题，请明确说明“根据我所掌握的信息，无法回答这个问题”，不要编造答案。
    3. 你的回答应该简洁明了。

    已知信息:
    {context}

    问题: {question}

    回答:
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)

    # 2. 构建 RAG 链 (使用 LCEL)
    # RunnablePassthrough() 会将 invoke 的输入原样传递下去。
    # 我们期望 invoke 的输入是一个字符串 (用户的问题)。
    # RunnableLambda(lambda x: x) 也是一种方式来明确表示传递输入。

    # retriever 希望输入是字符串（查询），输出是 Document 列表
    # format_docs 希望输入是 Document 列表，输出是字符串
    # rag_prompt 希望输入是字典 {"context": str, "question": str}
    # llm 希望输入是 PromptValue 或 Message 列表
    # StrOutputParser 希望输入是 AIMessage，输出是字符串

    rag_chain = (
        # RunnablePassthrough 将整个输入字典传递下去
        # 我们也可以只传递问题给 retriever，并将原始问题保留下来
        # 使用 RunnablePassthrough.assign 来创建新的键，或者使用字典推导来构建
        {
            "context": RunnableLambda(lambda user_input_dict: user_input_dict["question"]) | retriever_instance | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda user_input_dict: user_input_dict["question"])
        }
        | rag_prompt
        | llm_instance
        | StrOutputParser()
    )

    # 也可以这样写，如果 invoke 的输入直接是问题字符串：
    # rag_chain_alternative = (
    #     {"context": retriever_instance | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    #     | rag_prompt
    #     | llm_instance
    #     | StrOutputParser()
    # )
    # invoke 的时候直接 rag_chain_alternative.invoke("你的问题")

    # 3. 测试 RAG 链
    questions_to_ask = [
        "LangChain 的主要价值是什么？",
        "DeepSeek 是由哪家公司开发的？",
        "文本分割在处理长文本时有什么用？",
        "什么是机器学习？" # 这个问题应该在知识库中找不到答案
    ]

    for q in questions_to_ask:
        print(f"\n用户问题: {q}")
        try:
            # 我们链的输入现在期望是一个字典，因为我们是这样构造的
            answer = rag_chain.invoke({"question": q})
            # 如果用 rag_chain_alternative，则：
            # answer = rag_chain_alternative.invoke(q)
            print(f"AI 回答: {answer}")
        except Exception as e:
            import traceback
            print(f"  RAG 链执行失败: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    # 步骤 A: 初始化 LLM
    deepseek_llm = create_deepseek_llm()
    if not deepseek_llm:
        print("DeepSeek LLM 初始化失败，程序退出。")
        exit()

    # 步骤 B: 初始化嵌入模型
    embeddings = initialize_embedding_model()
    if not embeddings:
        print("嵌入模型初始化失败，程序退出。")
        exit()

    # 步骤 C: 加载向量数据库
    # 确保 "./my_chroma_data" 目录存在并且包含你之前创建并填充的数据
    # 如果这是第一次运行有关 Chroma 的脚本，你需要先运行一个脚本来创建和填充它。
    # 假设你已经有一个脚本（比如你之前提供的那个）可以创建 `my_chroma_data`。
    chroma_vector_store = load_chroma_vector_store(embeddings, persist_directory="./my_chroma_data")

    if not chroma_vector_store:
        print("未能加载向量数据库。请确保 './my_chroma_data' 目录已正确创建并包含数据。")
        print("你可以运行之前的 Chroma 创建脚本来生成它。")
        exit()

    # 步骤 D: 从 VectorStore 创建 Retriever
    # 你可以根据需要配置 search_kwargs
    knowledge_retriever = chroma_vector_store.as_retriever(
        search_kwargs={"k": 3} # 检索3个最相关的文档块
    )
    print(f"\n已从 Chroma 数据库创建 Retriever (k=3)。")

    # 步骤 E: 测试 RAG 链
    test_rag_chain(deepseek_llm, knowledge_retriever)

    # (你之前的 test_simple_sequential_chain() 可以保留或注释掉)
    # print("\n--- 分割线，运行旧的 SimpleSequentialChain 测试 ---")
    # test_simple_sequential_chain()