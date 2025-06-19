import numpy as np

# 文本A: "今天天气真好" 的向量
vector_a = np.array([0.9, 0.8, 0.1, 0.2])

# 文本B: "今天阳光明媚" 的向量
vector_b = np.array([0.8, 0.9, 0.1, 0.3])

# 文本C: "我喜欢吃披萨" 的向量
vector_c = np.array([0.1, 0.2, 0.9, 0.8])

a_b = vector_a @ vector_b
print(a_b) # 1.5100000000000002

a_c = vector_a @ vector_c
print(a_c) # 0.5000000000000001

b_c = np.dot(vector_b, vector_c)
print(b_c) # 0.5900000000000001