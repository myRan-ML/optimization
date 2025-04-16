import math

class Node:
    """
    表示计算图中的一个节点，每个节点对应一个操作（Op）的计算结果。
    """
    _global_id = 0  # 唯一标识符计数器
    
    def __init__(self, operation, inputs):
        """
        初始化一个节点，记录输入节点，操作符和梯度信息，并立即进行计算。
        :param operation: 该节点执行的操作（Op）。
        :param inputs: 输入节点列表或数值。
        """
        self.inputs = inputs  # 输入节点或数值
        self.operation = operation  # 操作符
        self.grad = 0.0  # 初始化梯度
        self.evaluate()  # 立即计算节点的值
        
        # 为该节点分配唯一ID
        self.id = Node._global_id
        Node._global_id += 1

        # 输出调试信息
        print(f"Eager execution: {self}")

    def _inputs_to_values(self):
        """
        将输入转换为数值，因为具体计算只能发生在数值上。
        :return: 转换后的数值列表。
        """
        return [input.value if isinstance(input, Node) else input for input in self.inputs]

    def evaluate(self):
        """ 计算节点的值 """
        self.value = self.operation.compute(self._inputs_to_values())

    def __repr__(self):
        return str(self)

    def __str__(self):
        """ 输出节点的信息 """
        return f"Node{self.id}: {self._inputs_to_values()} {self.operation.name()} = {self.value}, grad: {self.grad:.3f}"

class Operation:
    """
    所有计算操作的基类。每个操作产生一个新的Node并计算其结果。
    """
    def name(self):
        """ 返回操作的名称 """
        pass

    def __call__(self, *args):
        """ 创建新的节点，表示计算的结果 """
        pass

    def compute(self, inputs):
        """ 计算操作的结果 """
        pass

    def gradient(self, inputs, output_grad):
        """ 计算操作的梯度 """
        pass

class AddOperation(Operation):
    """ 加法操作 """
    def name(self):
        return "add"

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] + inputs[1]

    def gradient(self, inputs, output_grad):
        return [output_grad, output_grad]

class SubOperation(Operation):
    """ 减法操作 """
    def name(self):
        return "sub"

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] - inputs[1]

    def gradient(self, inputs, output_grad):
        return [output_grad, -output_grad]

class MulOperation(Operation):
    """ 乘法操作 """
    def name(self):
        return "mul"

    def __call__(self, a, b):
        return Node(self, [a, b])

    def compute(self, inputs):
        return inputs[0] * inputs[1]

    def gradient(self, inputs, output_grad):
        return [inputs[1] * output_grad, inputs[0] * output_grad]

class LnOperation(Operation):
    """ 自然对数操作 """
    def name(self):
        return "ln"

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, inputs):
        return math.log(inputs[0])

    def gradient(self, inputs, output_grad):
        return [1.0 / inputs[0] * output_grad]

class SinOperation(Operation):
    """ 正弦操作 """
    def name(self):
        return "sin"

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, inputs):
        return math.sin(inputs[0])

    def gradient(self, inputs, output_grad):
        return [math.cos(inputs[0]) * output_grad]

class IdentityOperation(Operation):
    """ 恒等操作（输入等于输出） """
    def name(self):
        return "identity"

    def __call__(self, a):
        return Node(self, [a])

    def compute(self, inputs):
        return inputs[0]

    def gradient(self, inputs, output_grad):
        return [output_grad]

class Executor:
    """
    计算图的执行器，负责按拓扑排序执行计算并进行反向传播（自动微分）。
    """
    def __init__(self, root_node):
        """
        初始化执行器，进行拓扑排序
        :param root_node: 计算图的根节点
        """
        self.topo_order = self._topological_sort(root_node)
        self.root_node = root_node

    def run(self):
        """
        执行计算图的前向计算。
        :return: 根节点的计算结果。
        """
        evaluated_nodes = set()  # 确保每个节点只计算一次
        print("\nEvaluation Order:")
        for node in self.topo_order:
            if node not in evaluated_nodes:
                node.evaluate()
                evaluated_nodes.add(node)
                print(f"Evaluating: {node}")
        return self.root_node.value

    def _dfs(self, node, topo_list):
        """ 深度优先遍历 """
        if node is None or not isinstance(node, Node):
            return
        for input_node in node.inputs:
            self._dfs(input_node, topo_list)
        topo_list.append(node)

    def _topological_sort(self, root):
        """ 拓扑排序 """
        topo_list = []
        self._dfs(root, topo_list)
        return topo_list

    def gradients(self):
        """
        执行反向传播，根据拓扑排序计算梯度。
        """
        reverse_order = list(reversed(self.topo_order))
        reverse_order[0].grad = 1.0  # 输出节点的梯度为1.0
        
        for node in reverse_order:
            grad = node.operation.gradient(node._inputs_to_values(), node.grad)
            # 将梯度累加到输入节点的梯度
            for input_node, g in zip(node.inputs, grad):
                if isinstance(input_node, Node):
                    input_node.grad += g

        print("\nAfter Autodiff:")
        for node in reverse_order:
            print(node)

# 示例：验证计算图
add_op, mul_op, ln_op, sin_op, sub_op, identity_op = AddOperation(), MulOperation(), LnOperation(), SinOperation(), SubOperation(), IdentityOperation()

x1, x2 = identity_op(2.0), identity_op(5.0)
y = sub_op(add_op(ln_op(x1), mul_op(x1, x2)), sin_op(x2))  # y = ln(x1) + x1 * x2 - sin(x2)

executor = Executor(y)
print(f"y = {executor.run():.3f}")
executor.gradients()  # 反向传播计算梯度

print(f"x1.grad = {x1.grad:.3f}")
print(f"x2.grad = {x2.grad:.3f}")
