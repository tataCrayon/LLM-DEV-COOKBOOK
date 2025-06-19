import os
from langchain_huggingface import HuggingFaceEmbeddings # 使用新的导入
from langchain_chroma import Chroma # 使用新的导入
from langchain_core.documents import Document # 如果需要手动创建文档用于测试

# --- 1. 初始化嵌入模型 (与之前相同) ---
def initialize_embedding_model():
    print("\n--- 初始化嵌入模型 ---")
    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {'device': 'cuda'} # 或者 'cuda' 如果 GPU 可用
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
        return None

# --- 2. 加载已持久化的 Chroma 数据库 ---
def load_chroma_vector_store(embedding_model, persist_directory="./my_chroma_data"):
    if not embedding_model:
        print("嵌入模型未初始化，无法加载 Chroma 数据库。")
        return None
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory): # 检查目录是否存在且不为空
        print(f"持久化目录 '{persist_directory}' 不存在或为空，无法加载。")
        print("请先运行创建数据库的脚本。")
        return None

    print(f"\n--- 从 '{persist_directory}' 加载已存在的 Chroma 数据库 ---")
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

# --- 3. 从 VectorStore 创建和使用 Retriever ---
def use_retriever_from_vector_store(vector_store_instance):
    if not vector_store_instance:
        print("向量数据库实例未提供，无法创建 Retriever。")
        return

    print("\n--- 从 VectorStore 创建 Retriever ---")

    # 3.1 基本的相似性搜索 Retriever
    # .as_retriever() 是最常用的方法
    # search_kwargs 可以用来配置底层的搜索行为，如 k (返回数量) 和 filter
    retriever_simple = vector_store_instance.as_retriever(
        search_kwargs={"k": 2} # 默认检索2个最相关的文档
    )
    print("已创建基本的相似性搜索 Retriever (k=2)。")

    query1 = "LangChain的价值主张是什么？"
    print(f"\n使用 Retriever 检索查询: '{query1}'")
    # Retriever 使用 .invoke() 方法 (LCEL 标准)
    # 或者旧版的 .get_relevant_documents()
    retrieved_docs1 = retriever_simple.invoke(query1)
    print(f"为查询 '{query1}' 检索到 {len(retrieved_docs1)} 个文档:")
    for i, doc in enumerate(retrieved_docs1):
        print(f"  文档 {i+1} (来自: {doc.metadata.get('source', 'N/A')}): '{doc.page_content[:100]}...'")

    # 3.2 Retriever with Metadata Filtering
    retriever_filtered = vector_store_instance.as_retriever(
        search_kwargs={
            "k": 1,
            "filter": {"category": "llm_provider"} # 只检索 category 为 'llm_provider' 的文档
        }
    )
    print("\n已创建带元数据过滤的 Retriever (k=1, category='llm_provider')。")

    query2 = "DeepSeek公司是做什么的？" # 这个查询更可能匹配 'llm_provider' 类别
    print(f"\n使用过滤 Retriever 检索查询: '{query2}'")
    retrieved_docs2 = retriever_filtered.invoke(query2)
    print(f"为查询 '{query2}' (过滤后) 检索到 {len(retrieved_docs2)} 个文档:")
    for i, doc in enumerate(retrieved_docs2):
        print(f"  文档 {i+1} (元数据: {doc.metadata}): '{doc.page_content[:100]}...'")


    # 3.3 Retriever with MMR (Maximal Marginal Relevance) for diversity
    # MMR 尝试在与查询相关的同时，最大化结果集的多样性。
    # 这需要向量数据库支持 MMR 搜索。Chroma 支持它。
    retriever_mmr = vector_store_instance.as_retriever(
        search_type="mmr", # 指定搜索类型为 MMR
        search_kwargs={
            "k": 3,             # 获取的文档总数
            "fetch_k": 10,      # MMR 算法从多少个初始最相似的文档中选择 (应 >= k)
            "lambda_mult": 0.5  # 控制多样性与相关性的平衡 (0-1之间，0.5是平衡，接近1更看重相关性，接近0更看重多样性)
        }
    )
    print("\n已创建带 MMR 的 Retriever (k=3, fetch_k=10, lambda_mult=0.5)。")
    query3 = "LangChain有什么用？" # 一个比较泛的问题
    print(f"\n使用 MMR Retriever 检索查询: '{query3}'")
    retrieved_docs3 = retriever_mmr.invoke(query3)
    print(f"为查询 '{query3}' (MMR) 检索到 {len(retrieved_docs3)} 个文档:")
    for i, doc in enumerate(retrieved_docs3):
        print(f"  文档 {i+1} (元数据: {doc.metadata}): '{doc.page_content[:100]}...'")

    # 3.4 Retriever with Similarity Score Threshold
    # 只返回相似度得分达到一定阈值的文档。
    # 注意：得分的含义和阈值的设置取决于向量数据库和相似度度量。
    # Chroma 使用距离（默认 L2），所以得分越小越相似。
    # LangChain 的 SimilarityThresholdRetriever 是一个更通用的包装器，
    # 但 VectorStoreRetriever 也可以通过 search_kwargs 来尝试实现。
    # 对于 Chroma，你可能需要直接用 similarity_search_with_score 然后自己过滤，
    # 或者看 as_retriever 是否能直接支持 score_threshold。
    # 更稳妥的方式是使用 `SimilarityThresholdRetriever` 包装一个 `VectorStore`
    from langchain.retrievers import SelfQueryRetriever # 只是为了展示，下面用一个更简单的
    from langchain.retrievers import ContextualCompressionRetriever # 只是为了展示
    # 对于基于得分阈值的检索，通常 VectorStoreRetriever 本身通过 search_kwargs 可能不直接支持 "score_threshold"
    # 我们通常会用 `similarity_search_with_relevance_scores` (如果可用) 或 `similarity_search_with_score` 然后手动过滤
    # 或者使用专门的 Retriever 包装器。

    # 让我们演示一个简单的基于 search_kwargs 的 k 和 filter
    # 如果想用得分阈值，Chroma 的 as_retriever 可能通过 search_type="similarity_score_threshold" 支持，
    # 或者需要更底层的配置。
    # LangChain 的 VectorStoreRetriever 文档说明 search_type 可以是 'similarity_score_threshold'
    try:
        retriever_score_thresh = vector_store_instance.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.3 # 这里的 score_threshold 意义取决于 Chroma 的配置和相似度函数
                                        # 对于 Chroma 默认的 L2 距离，小的表示更相似。
                                        # 所以 0.3 可能是一个相对严格的（非常相似的）阈值。
                                        # 如果是余弦相似度，则可能需要 0.8 或更高。
                                        # 你需要试验这个值。
            }
        )
        print(f"\n已创建带相似度得分阈值的 Retriever (k=3, score_threshold=0.3，具体效果依赖 Chroma 配置)。")
        query4 = "介绍一下文本分割"
        print(f"\n使用得分阈值 Retriever 检索查询: '{query4}'")
        retrieved_docs4 = retriever_score_thresh.invoke(query4)
        print(f"为查询 '{query4}' (得分阈值) 检索到 {len(retrieved_docs4)} 个文档:")
        for i, doc in enumerate(retrieved_docs4):
            print(f"  文档 {i+1} (元数据: {doc.metadata}): '{doc.page_content[:100]}...'")
    except Exception as e:
        print(f"创建或使用得分阈值 Retriever 时出错 (这可能表示 Chroma 的 LangChain 包装器对该 search_type 的支持有限或配置不当): {e}")


if __name__ == '__main__':
    # 1. 初始化嵌入模型
    embeddings = initialize_embedding_model()

    if embeddings:
        # 2. 加载向量数据库
        # 确保你的 "./my_chroma_data" 目录存在并且包含之前创建的数据
        # 如果是第一次运行，你需要先运行创建和填充数据库的脚本
        chroma_vector_store = load_chroma_vector_store(embeddings, persist_directory="./my_chroma_data")

        if chroma_vector_store:
            # 3. 使用 Retriever
            use_retriever_from_vector_store(chroma_vector_store)
            print("\nRetriever 演示完成。")
            # 下一步就是将这些 Retriever 集成到 RAG 链中，与 LLM 结合起来！
        else:
            print("未能加载向量数据库，无法继续进行 Retriever 演示。")
    else:
        print("嵌入模型初始化失败，无法继续。")