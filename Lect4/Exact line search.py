import numpy as np

# f(x) = 0.5 * x^T * Q * x + c^T * x
def f(x, Q, c):
    f_x = 0.5 * x.T @ Q @ x + c @ x
    return f_x

# ∇f(x) = Q * x + c
def gradient(x, Q, c):
    return Q @ x + c

# 精确线索所步长
def compute_alpha(x, Q, c):
    grad = gradient(x, Q, c)

    numerator = grad.T @ grad
    denominator = grad.T @ Q @ grad
    return grad, numerator / denominator

Q = np.array([[2.0, 1.0, 0], [1.0, 2.0, 1.0], [0, 1.0, 2.0]])
x_init = np.array([8.0, 11.0, 7.0]) 
c = np.array([2.0, 5.0, 7.0])
iters_num = 10000              #迭代次数
learning_rate = 0.01           #固定步长
epsilon = 1e-6                 #终止条件

x_1 = x_init.copy()
x_2 = x_init.copy()

for i in range(iters_num):
    grad = gradient(x_1, Q, c)
    norm_grad = np.linalg.norm(grad)
    if norm_grad < epsilon:
        print(f"固定步长梯度下降法在第 {i} 次迭代后收敛。")
        break
    x_1 -= learning_rate * grad
print("固定步长梯度下降法的最终结果:", x_1)

for i in range(iters_num):
    grad, alpha = compute_alpha(x_2, Q, c)
    norm_grad = np.linalg.norm(grad)
    if norm_grad < epsilon:
        print(f"自适应步长梯度下降法在第 {i} 次迭代后收敛。")
        break
    x_2 -= alpha * grad
print("自适应步长梯度下降法的最终结果:", x_2)

# 计算 Q 的逆矩阵
Q_inv = np.linalg.inv(Q)
# 计算理论最优解
x_optimal = -np.dot(Q_inv, c)

print("理论上的最优点坐标:", x_optimal)