import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. 嵌入 (Embeddings) ---
from langchain_huggingface import HuggingFaceEmbeddings

# --- 2. 向量数据库 (Vector Stores) ---
from langchain_chroma import Chroma


# (复用之前的示例文本和文档对象)
long_text_example = """LangChain 是一个用于开发由语言模型驱动的应用程序的框架。
它使得应用程序能够：
- 具有上下文感知能力：将语言模型连接到上下文来源（提示指令、少量示例、需要响应的内容等）。
- 具有推理能力：依赖语言模型进行推理（关于如何根据提供的上下文进行操作、何时进行操作等）。

LangChain 的主要价值主张是：
1. 组件化：LangChain 提供了模块化的抽象，用于构建语言模型应用程序所需的组件。
2. 用例驱动的链：LangChain 还提供了特定用例的链，这些链是预先构建的组件组合。

文本分割是处理长文本时的重要步骤。选择合适的分割器和参数对于后续的嵌入和检索效果至关重要。
嵌入模型将文本转换为数字向量，捕捉其语义。相似的文本具有相似的向量。
向量数据库用于存储这些向量并进行高效的相似性搜索。Chroma 是一个流行的选择。
"""

sample_document = Document(page_content=long_text_example, metadata={"source": "manual_example", "category": "framework_intro", "doc_id": "doc1"})

another_document_content = """DeepSeek 是由深度求索公司开发的一系列大型语言模型。
它包括了代码模型和通用对话模型。DeepSeek Coder 在代码生成任务上表现优异。
DeepSeek-LLM 67B 是一个强大的通用基础模型。
这些模型旨在推动人工智能领域的发展，并为开发者提供强大的工具。
深度求索致力于开源其研究成果，促进社区合作。
"""
another_document = Document(page_content=another_document_content, metadata={"source": "deepseek_info", "category": "llm_provider", "doc_id": "doc2"})

all_documents = [sample_document, another_document]


# --- 步骤 1: 分割文档 ---
def get_document_chunks(documents_to_split):
    print("\n--- 步骤 1: 分割文档 ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents_to_split)
    print(f"文档被分割成了 {len(chunks)} 个块。")
    return chunks

# --- 步骤 2: 初始化嵌入模型 ---
def initialize_embedding_model():
    print("\n--- 步骤 2: 初始化嵌入模型 ---")
    model_name = "BAAI/bge-small-zh-v1.5"
    model_kwargs = {'device': 'cuda'}
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


# --- 步骤 3 & 4: 使用 Chroma 创建向量数据库并进行相似性搜索 ---
def create_chroma_vector_store_and_search(document_chunks, embedding_model, persist_directory="./chroma_db_store"):
    if not document_chunks or not embedding_model:
        print("文档块或嵌入模型未准备好，跳过向量数据库创建和搜索。")
        return None

    print("\n--- 步骤 3: 从文档块创建 Chroma 向量数据库 ---")
    # 如果 persist_directory 存在且包含数据，Chroma 会尝试加载它。
    # 否则，它会创建一个新的数据库并持久化到该目录。
    # 如果只想在内存中运行（不持久化），可以不指定 persist_directory。
    # vector_store = Chroma.from_documents(documents=document_chunks, embedding=embedding_model) # 内存模式

    try:
        # 确保目录存在，如果需要
        if persist_directory and not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"创建持久化目录: {persist_directory}")

        vector_store = Chroma.from_documents(
            documents=document_chunks, # 文档块列表
            embedding=embedding_model,   # 嵌入函数
            persist_directory=persist_directory, # 持久化存储的目录
            collection_name="my_knowledge_base" # （可选）给集合一个名字
        )
        print(f"Chroma 向量数据库创建/加载成功，并持久化到 '{persist_directory}'。")
        # 调用 persist() 确保数据被写入磁盘 (对于某些版本的Chroma可能是必要的，或者在特定操作后)
        # vector_store.persist() # 在最新版本中，from_documents 时指定 persist_directory 通常会自动处理

    except Exception as e:
        print(f"创建 Chroma 向量数据库时出错: {e}")
        return None

    print("\n--- 步骤 4: 进行相似性搜索 (使用 Chroma) ---")
    query1 = "LangChain的价值主张是什么？"
    print(f"\n查询 1: '{query1}'")
    results1 = vector_store.similarity_search(query1, k=2)
    for i, doc in enumerate(results1):
        print(f"  结果 {i+1} (来自: {doc.metadata.get('source', 'N/A')}, 元数据: {doc.metadata}):")
        print(f"    内容: '{doc.page_content[:150]}...'")

    query2 = "DeepSeek 是什么？"
    print(f"\n查询 2: '{query2}'")
    results2 = vector_store.similarity_search_with_score(query2, k=2)
    for i, (doc, score) in enumerate(results2):
        # Chroma 的分数通常是距离 (如 L2 距离或余弦距离的相反数)，越小越相似
        # 或者如果配置为余弦相似度，则越大越相似。默认情况下，LangChain 的 Chroma 包装器
        # 返回的 score 是距离，所以越小越好。
        print(f"  结果 {i+1} (来自: {doc.metadata.get('source', 'N/A')}, 得分: {score:.4f}, 元数据: {doc.metadata}):")
        print(f"    内容: '{doc.page_content[:150]}...'")

    # 使用元数据过滤进行搜索
    query_filtered = "LangChain的组件有哪些？"
    print(f"\n带元数据过滤的查询: '{query_filtered}' (只在 'framework_intro' 类别中搜索)")
    try:
        # Chroma 的过滤语法是使用 `where` 子句，类似于 MongoDB
        # 对于简单的等值匹配，可以这样写：
        results_filtered = vector_store.similarity_search(
            query_filtered,
            k=2,
            filter={"category": "framework_intro"} # LangChain 抽象层面的 filter
            # 或者使用 Chroma 原生的 where (更灵活，但可能需要通过 search_kwargs 传递)
            # search_kwargs={'where': {'category': 'framework_intro'}}
        )
        if results_filtered:
            for i, doc in enumerate(results_filtered):
                print(f"  过滤结果 {i+1} (元数据: {doc.metadata}):")
                print(f"    内容: '{doc.page_content[:150]}...'")
        else:
            print("  没有找到符合过滤条件的文档。")
    except Exception as e:
        print(f"  带元数据过滤的搜索失败: {e}")


    # 将 VectorStore 转换为 Retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3, "filter": {"category": "llm_provider"}}
    )
    print(f"\n查询 (使用 Retriever, 过滤类别 'llm_provider'): '{query2}'") # 用 query2 测试
    retrieved_docs = retriever.invoke(query2)
    for i, doc in enumerate(retrieved_docs):
        print(f"  Retrieved Doc {i+1} (元数据: {doc.metadata}): '{doc.page_content[:100]}...'")

    return vector_store

# --- 如何加载已持久化的 Chroma 数据库 ---
def load_persisted_chroma_db(embedding_model, persist_directory="./chroma_db_store"):
    if not embedding_model:
        print("嵌入模型未初始化，无法加载 Chroma 数据库。")
        return None
    if not os.path.exists(persist_directory):
        print(f"持久化目录 '{persist_directory}' 不存在，无法加载。")
        return None

    print(f"\n--- 尝试从 '{persist_directory}' 加载已存在的 Chroma 数据库 ---")
    try:
        loaded_vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model, # 必须提供相同的嵌入函数
            collection_name="my_knowledge_base" # 如果创建时指定了，加载时也需要
        )
        print("Chroma 数据库加载成功。")

        # 测试一下加载的数据库
        query_test = "什么是 LangChain 组件化？"
        print(f"\n在加载的数据库中测试查询: '{query_test}'")
        results_test = loaded_vector_store.similarity_search(query_test, k=1)
        if results_test:
            print(f"  结果 (元数据: {results_test[0].metadata}): '{results_test[0].page_content[:100]}...'")
        else:
            print("  未找到结果。")
        return loaded_vector_store
    except Exception as e:
        print(f"加载 Chroma 数据库时出错: {e}")
        print("确保持久化目录存在且包含有效的 Chroma 数据，并且嵌入函数与创建时一致。")
        return None

if __name__ == '__main__':
    doc_chunks = get_document_chunks(all_documents)
    embeddings_model = initialize_embedding_model()

    if embeddings_model:
        # 首次运行时会创建并持久化
        chroma_db = create_chroma_vector_store_and_search(doc_chunks, embeddings_model, persist_directory="./my_chroma_data")

        if chroma_db:
            print("\nChromaDB 创建和检索演示完成。")
            # 你可以删除 ./my_chroma_data 目录再运行，它会重新创建
            # 或者注释掉 create_chroma_vector_store_and_search，然后取消下面的注释来测试加载

        # --- 测试加载持久化的数据库 ---
        # (确保上面的 persist_directory 和这里的路径一致，比如都用 "./my_chroma_data")
        print("\n--- 第二次运行或在不同脚本中，可以尝试加载 ---")
        loaded_db = load_persisted_chroma_db(embeddings_model, persist_directory="./my_chroma_data")
        if loaded_db:
            # 你现在可以使用 loaded_db 进行更多查询了
            query_after_load = "深度求索公司是做什么的？"
            print(f"\n用加载的DB查询: '{query_after_load}'")
            search_results = loaded_db.similarity_search(query_after_load, k=1)
            if search_results:
                 print(f"  结果 (元数据: {search_results[0].metadata}): '{search_results[0].page_content[:100]}...'")

    else:
        print("由于嵌入模型初始化失败，无法继续进行向量数据库操作。")