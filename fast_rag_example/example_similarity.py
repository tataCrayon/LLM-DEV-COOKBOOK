from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

# 加载模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 输入句子
sentences = ["今天天气很好。", "天气晴朗适合户外活动。"]

# 生成嵌入（默认归一化）
embeddings = model.encode(sentences, normalize_embeddings=True)

# 计算点积
dot_product = torch.dot(torch.tensor(embeddings[0]), torch.tensor(embeddings[1]))

# 计算余弦相似度
cosine_similarity = util.cos_sim(embeddings, embeddings)[0][1]

# 计算欧氏距离
euclidean_distance = np.linalg.norm(embeddings[0] - embeddings[1])

print(f"点积: {dot_product:.4f}")
print(f"余弦相似度: {cosine_similarity:.4f}")
print(f"欧氏距离: {euclidean_distance:.4f}")