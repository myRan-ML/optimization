# Lect3 微积分基础

### 1. 向量微分与求导
- **可微函数定义**：线性近似与梯度向量 $\nabla f(x)$。
- **方向导数**：沿方向 $d$ 的导数 $\frac{\partial f}{\partial d} = \nabla f(x)^T d$。
- **微分运算**：利用迹（trace）简化计算，如 $df = \text{tr}(\nabla f(x)^T dx)$。

### 2. 映射微分与链式法则
- **Jacobi 矩阵**：映射 $F: \mathbb{R}^n \to \mathbb{R}^m$ 的微分
$\nabla_{\mathbf{x}} F(\mathbf{x})=\frac{\partial F}{\partial \mathbf{x}}=\left(\begin{array}{ccc}\frac{\partial f_{1}}{\partial x_{1}} & \cdots & \frac{\partial f_{m}}{\partial x_{1}} \\\vdots & \vdots & \vdots \\\frac{\partial f_{1}}{\partial x_{n}} & \cdots & \frac{\partial f_{m}}{\partial x_{n}}\end{array}\right)$
- **链式法则**：复合函数梯度 $\nabla (f \circ g) = \frac{\partial g}{\partial x} \nabla f(g(x))$。

### 3. 矩阵函数求导
- **迹法**：对于矩阵变量函数，如果微分能够写成 \( df = \text{tr}(\square^T d\mathbf{x}) \)，那么 \(\square\) 就是函数对变量的梯度，即 \(\square = \nabla f(\mathbf{X}) = \frac{\partial f}{\partial \mathbf{X}}\)。

- **高阶应用**：矩阵分解与对数行列式梯度 $\nabla \log \det X = X^{-T}$。

### 4. 微分中值定理
- **Taylor 展开**：多变量函数的二阶近似：
  $$f(x+p) = f(x) + \nabla f(x)^T p + \frac{1}{2} p^T \nabla^2 f(x+\xi p) p.$$

---

## 常见公式示例

| 函数类型 | 梯度 |
| --- | --- |
| **线性函数**：\( \mathbf{a}^T \mathbf{x} \) | \( \nabla f = \mathbf{a} \) |
| **对称二次型**：\( \mathbf{x}^T A \mathbf{x} \) | \( \nabla f = 2A\mathbf{x} \) |
| **一般二次型**：\( \mathbf{x}^T B \mathbf{x} \) | \( \nabla f = (B + B^T)\mathbf{x} \) |
| **范数平方**：\( \|\mathbf{x}\|^2 \) | \( \nabla f = 2\mathbf{x} \) |
| **线性变换范数**：\( \|A\mathbf{x} - \mathbf{b}\|^2 \) | \( \nabla f = 2A^T(A\mathbf{x} - \mathbf{b}) \) |
| **复合函数**：\( g(A\mathbf{x} + \mathbf{b}) \) | \( \nabla f = A^T \nabla g(A\mathbf{x} + \mathbf{b}) \) |
