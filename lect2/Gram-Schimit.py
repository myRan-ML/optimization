import numpy as np

def gram_schmidt(x_vectors):
    """
    对线性无关的向量组进行Gram-Schmidt正交化。
    
    参数:
    x_vectors (list of numpy.ndarray): 输入向量列表，每个元素为numpy数组。
    
    返回:
    list of numpy.ndarray: 正交化后的向量列表。
    """
    u_vectors = []
    for i in range(len(x_vectors)):
        xi = x_vectors[i].astype(float)  # 转换为浮点数以确保精度
        ui = xi.copy()
        for j in range(i):
            uj = u_vectors[j]
            # 计算投影系数
            proj_coeff = np.dot(xi, uj) / np.dot(uj, uj)
            # 减去投影分量
            ui -= proj_coeff * uj
        u_vectors.append(ui)
    return u_vectors

# 示例测试
if __name__ == "__main__":
    # 定义输入向量（线性无关）
    x1 = np.array([1, 1])
    x2 = np.array([1, 0])
    vectors = [x1, x2]
    
    # 执行Gram-Schmidt正交化
    orthogonal_vectors = gram_schmidt(vectors)
    
    # 输出结果
    print("正交化后的向量：")
    for i, u in enumerate(orthogonal_vectors):
        print(f"u{i+1} = {u}")
        
    # 验证正交性
    print("\n验证正交性：")
    for i in range(len(orthogonal_vectors)):
        for j in range(i+1, len(orthogonal_vectors)):
            dot = np.dot(orthogonal_vectors[i], orthogonal_vectors[j])
            print(f"u{i+1}·u{j+1} = {dot:.2f}")