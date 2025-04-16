import torch

# -------------------- scalar ------------------------
# 定义输入并启用梯度追踪
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
print(f"x and y are scalar: x = {x},y = {y}")

# 前向计算
z = x * y + 1.0 # 定义函数

# 反向传播（假设z是标量）
z.backward()

# 输出梯度
print("dz/dx:", x.grad)  # 3.0 (y的值)
print("dz/dy:", y.grad)  # 2.0 (x的值)

print("\n")
# -------------------- tensor ------------------------
x = torch.arange(4.0)
print(f"x is a tensor: {x}")

x.requires_grad_(True) # 等价于x=torch.arange(4.0,requires_grad=True)  
x.grad # 默认值是None

y = 2 * torch.dot(x, x)
print(f"y is a scalar: {y}")

y.backward()

# 由线性代数的知识，x.grad == 4 * x
print(x.grad == 4 * x) # 验证通过
print(x.grad == 3 * x) # 验证不通过 
print("x.grad:")
print(x.grad)
print("\n")


print("if you don't let x.grad zero:")
y = x.sum()  # x的各分量相加得到标量，赋值给y
print(y)
y.backward()  
print(f"x.grad = {x.grad}")


print("\n")
print("if you do let x.grad zero:")
# 在默认情况下,PyTorch会累积梯度,我们需要清除之前的值  
x.grad.zero_() # 梯度清零
y = x.sum()  # x的各分量相加得到标量，赋值给y
print(y)
y.backward()  
print(f"x.grad = {x.grad}")
