from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# --- 准备数据 (沿用之前的代码) ---
model_name = 'BAAI/bge-small-zh-v1.5'
model = SentenceTransformer(model_name)

text_chunks = [
    "RAG的核心思想是开卷考试。",
    "RAG是一种结合了检索与生成的先进技术。",
    "今天天气真好，万里无云。",
    "在Java编程中，垃圾回收机制是JVM的重要组成部分。",
    "向量数据库专门用于高效地存储和检索高维数据。"
]

vectors = model.encode(text_chunks)

# --- 存储到ChromaDB ---

# 1. 创建一个ChromaDB客户端。
#    我们使用持久化存储，将数据保存在磁盘上的 "my_rag_db" 目录中。
client = chromadb.PersistentClient(path="my_rag_db")

# 2. 创建一个“集合”(Collection)，类似于SQL中的“表”。
#    如果集合已存在，可以先删除再创建，或直接获取。
collection_name = "my_first_collection"
try:
    client.delete_collection(name=collection_name)
except Exception:
    pass # 如果集合不存在会报错，我们忽略它
collection = client.create_collection(name=collection_name)

# 3. 将数据添加到集合中。
#    我们需要提供：
#    - documents: 原始文本块列表。
#    - embeddings: 对应的向量列表。
#    - ids: 每个文本块的唯一ID。
collection.add(
    embeddings=vectors,
    documents=text_chunks,
    ids=[f"chunk_{i}" for i in range(len(text_chunks))]
)

print("数据已成功存入ChromaDB！")


# --- 查询向量数据库 ---

# 4. 准备一个用户问题，并将其向量化
user_query = "什么是RAG技术？"
query_vector = model.encode(user_query)

# 5. 使用 .query() 方法进行查询
#    - query_embeddings: 查询向量。
#    - n_results: 希望返回的最相似的结果数量。
results = collection.query(
    query_embeddings=[query_vector.tolist()], # ChromaDB期望一个列表的列表
    n_results=2
)

# 6. 观察查询结果
print("\n--- 查询结果 ---")
print(results)