import numpy as np

# 1. 内积与范数计算
# 定义两个向量
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# (1)计算内积（点积）
inner_product = np.dot(v1, v2)
print("向量内积:", inner_product)  # 输出: 32

# (2)计算范数
l2_norm = np.linalg.norm(v1)       # L2范数（默认）
l1_norm = np.linalg.norm(v1, ord=1)  # L1范数
print(f"L2范数: {l2_norm:.2f}, L1范数: {l1_norm}")


# 2. 矩阵分解
# 定义矩阵
A = np.array([[4, 1], 
              [1, 3]])

# (1) 特征分解 (Eigen Decomposition)
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\n特征值:", eigenvalues)
print("特征向量矩阵:\n", eigenvectors)

# (2) QR分解
Q, R = np.linalg.qr(A)
print("\nQR分解:")
print("正交矩阵 Q:\n", Q)
print("上三角矩阵 R:\n", R)

# (3) 奇异值分解 (SVD)
U, S, VT = np.linalg.svd(A)
print("\n奇异值分解:")
print("左奇异矩阵 U:\n", U)
print("奇异值 S:", S)
print("右奇异矩阵 V^T:\n", VT)

# (4) Cholesky分解（需矩阵为对称正定）
L = np.linalg.cholesky(A)
print("\nCholesky分解的下三角矩阵 L:\n", L)


# 3. 验证分解正确性
# (1)验证特征分解: A = V diag(λ) V⁻¹
reconstructed_A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
print("\n特征分解重构误差:", np.linalg.norm(A - reconstructed_A))

# (2)验证QR分解: A = QR
print("QR分解重构误差:", np.linalg.norm(A - Q @ R))

# (3)验证Cholesky分解: A = LL^T
print("Cholesky重构误差:", np.linalg.norm(A - L @ L.T))