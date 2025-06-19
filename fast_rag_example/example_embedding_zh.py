# 1. 从库中导入 SentenceTransformer 类
from sentence_transformers import SentenceTransformer
import numpy as np
from sentence_transformers import util
# --- 向量化部分 ---

# 2. 加载本地模型。
#    第一次运行时，它会自动从Hugging Face下载模型文件到您的本地缓存中。
#    这可能需要一些时间，取决于您的网络。
model_name = 'BAAI/bge-small-zh-v1.5'
print(f"正在加载本地模型: {model_name}...")
model = SentenceTransformer(model_name)
print("模型加载完成。")

# 3. 准备一些待转换的文本块
text_chunks = [
    "RAG的核心思想是开卷考试。",
    "RAG是一种结合了检索与生成的先进技术。", # 与上一句意思相近
    "今天天气真好，万里无云。" # 与前两句意思完全不同
]

# 4. 使用 model.encode() 方法进行向量化。
#    它会返回一个Numpy数组的列表。
vectors = model.encode(text_chunks)


# --- 观察与计算部分 ---

# 5. 观察向量的形状 (shape)
print("\n--- 向量信息 ---")
# (句子数量, 每个向量的维度)
print(f"生成向量的形状: {vectors.shape}") 

# 6. 计算相似度：我们来计算第一个句子和另外两个句子的相似度
#    我们将使用余弦相似度，这是衡量向量方向一致性的标准方法。
#    公式: (A·B) / (||A|| * ||B||)

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1) # linalg.norm 计算向量的模长
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

similarity_1_vs_2 = cosine_similarity(vectors[0], vectors[1])
similarity_1_vs_3 = cosine_similarity(vectors[0], vectors[2])

print("\n--- 相似度计算结果 ---")
print(f"句子1 vs 句子2的相似度: {similarity_1_vs_2:.4f}") #  (语义相近) 
print(f"句子1 vs 句子3的相似度: {similarity_1_vs_3:.4f}") #  (语义无关)

similarities = util.cos_sim(vectors, vectors)
print(f"句子1 vs 句子2: {similarities[0][1]:.4f}")
print(f"句子1 vs 句子3: {similarities[0][2]:.4f}")