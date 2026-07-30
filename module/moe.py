"""
@function: moe结构
@author: jiazhuo

This module implements various Mixture of Experts (MoE) architectures including:
- MMOE
- CGC
- AdaTT
"""
import tensorflow as tf
from tensorflow.keras import regularizers
from logger import logger


class MMOE(tf.keras.layers.Layer):
    def __init__(self, num_experts, num_tasks, expert_dims, gate_dims, use_bn, l2_reg, training=False, **kwargs):
        super(MMOE, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_dims = expert_dims  # 专家网络隐藏层维度列表
        self.gate_dims = gate_dims      # 门控网络隐藏层维度列表
        self.use_bn = use_bn
        self.l2_reg = l2_reg
        self.training = training

    def build(self, input_shape):
        # 为每个专家创建多层DNN
        self.expert_networks = []
        for i in range(self.num_experts):
            expert_layers = []
            for j, dim in enumerate(self.expert_dims):
                expert_layers.append(
                    tf.keras.layers.Dense(
                        units=dim,
                        activation=tf.nn.swish,
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name='expert_{}_layer_{}'.format(i, j)
                    )
                )

                if self.use_bn:
                    expert_layers.append(tf.keras.layers.BatchNormalization())
                expert_layers.append(tf.keras.layers.Dropout(0.5))

            expert = tf.keras.Sequential(expert_layers, name='expert_{}'.format(i))
            self.expert_networks.append(expert)

        # 为每个任务创建独立的门控网络（多层）
        self.task_gates = []
        for i in range(self.num_tasks):
            gate_layers = []
            for j, dim in enumerate(self.gate_dims):
                gate_layers.append(
                    tf.keras.layers.Dense(
                        units=dim,
                        activation=tf.nn.swish,
                        kernel_regularizer=regularizers.l2(self.l2_reg),
                        name='task_{i}_gate_hidden_{j}'.format(i=i, j=j)
                    )
                    )
            # 最后一层输出 num_experts（softmax 归一化）
            gate_layers.append(
                tf.keras.layers.Dense(
                    units=self.num_experts,
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name='task_{i}_gate_output'.format(i=i),
                )
            )
            gate_network = tf.keras.Sequential(gate_layers, name='task_{}_gate'.format(i))
            self.task_gates.append(gate_network)

    def call(self, expert_input, gate_input_list, lr, fm, step=None):
        # expert输出
        expert_outputs_list = [expert(expert_input, training=self.training) for expert in self.expert_networks]
        expert_outputs_list_new = []
        for i in range(self.num_tasks):
            expert_outputs_list_new.append(tf.concat([lr, fm, expert_outputs_list[i]], axis=1))
        expert_outputs = tf.stack(expert_outputs_list_new, axis=1)  # [B,num_experts,expert_dim]

        task_outputs = []
        for i in range(self.num_tasks):
            gate_output = self.task_gates[i](gate_input_list[i], training=self.training)  # [B,num_experts]
            gate_outputs = tf.expand_dims(gate_output, axis=-1)  # [B,num_experts，1]
            mixed_output = tf.reduce_sum(expert_outputs * gate_outputs, axis=1)  # [B,expert_dim]
            task_outputs.append(mixed_output)

        return task_outputs

""""
*******PLE*******
"""

class PLEModel(tf.keras.layers.Layer):
    """
    PLE模型
    输出：num_tasks 个任务特征
    """
    def __init__(self,
                 num_experts_shared=4,
                 num_experts_specific=1,
                 expert_units=[1024,512,128],
                 gate_units=[128],
                 num_tasks=4,
                 num_levels=2,
                 expert_dropout=0.0,
                 use_bn=True):  # 专家网络的dropout
        super().__init__()
        # 核心参数
        self.num_experts_shared = num_experts_shared #共享expert 个数
        self.num_experts_specific = num_experts_specific #专属expert个数
        self.expert_units = expert_units  # 专家网络的隐藏层维度列表
        self.gate_units=gate_units        #门控网络隐藏层纬度列表
        self.num_tasks = num_tasks
        self.num_levels = num_levels
        self.expert_dropout = expert_dropout
        self.use_bn=use_bn

        # 1. 初始化所有CGC层的专家网络和门控网络
        self.cgc_layers = []
        for level in range(num_levels):
            is_last_layer = (level == num_levels - 1)
            self.cgc_layers.append({
                "is_last_layer": is_last_layer,
                # 特定任务专家（num_tasks × num_experts_specific 个）
                "experts_specific": [[
                    MultiLayerPerceptron(hidden_dims=expert_units, dropout=expert_dropout,use_bn=self.use_bn)
                    for _ in range(num_experts_specific)
                ] for _ in range(num_tasks)],
                # 共享专家（num_experts_shared 个）
                "experts_shared": [
                    MultiLayerPerceptron(hidden_dims=expert_units, dropout=expert_dropout,use_bn=self.use_bn)
                    for _ in range(num_experts_shared)
                ],
                # 任务门控（num_tasks 个）
                "task_gates": [
                    MultiLayerPerceptron(
                        hidden_dims=self.gate_units,
                        dropout=0.0,
                        output_layer=True,
                        output_activation='softmax',
                        output_dim=self.num_experts_shared+self.num_experts_specific
                    ) for _ in range(num_tasks)
                ],
                # 共享门控（非最后一层才有）
                "shared_gate": None if is_last_layer else MultiLayerPerceptron(
                        hidden_dims=self.gate_units,
                        dropout=0.0,
                        output_layer=True,
                        output_activation='softmax',
                        output_dim=self.num_experts_shared+self.num_experts_specific*num_tasks
                    )
            })

    def cgc_forward(self, inputs, cgc_config,training=None):
        """CGC层前向传播（核心逻辑）"""
        num_tasks = self.num_tasks
        num_experts_shared = self.num_experts_shared
        num_experts_specific = self.num_experts_specific
        is_last_layer = cgc_config["is_last_layer"]

        # ========== 1. 特定任务专家前向 ==========
        expert_outputs_specific = []
        for i in range(num_tasks):
            task_expert_outputs = [
                cgc_config["experts_specific"][i][j](inputs[i],training=training)  # 第i个任务的第j个特定专家
                for j in range(num_experts_specific)
            ]
            # 堆叠：(batch_size, num_experts_specific, expert_units)
            task_expert_outputs = tf.stack(task_expert_outputs, axis=1)
            expert_outputs_specific.append(task_expert_outputs)

        # ========== 2. 共享专家前向 ==========
        expert_outputs_shared = [
            cgc_config["experts_shared"][k](inputs[-1],training=training)  # 共享专家输入是最后一个input（共享分支）
            for k in range(num_experts_shared)
        ]
        # 堆叠：(batch_size, num_experts_shared, expert_units)
        expert_outputs_shared = tf.stack(expert_outputs_shared, axis=1)

        # ========== 3. 任务门控加权 ==========
        task_outputs = []
        for i in range(num_tasks):
            # 门控输出：reshape为 (batch_size, num_experts_shared + num_experts_specific)
            gate_output = cgc_config["task_gates"][i](inputs[i],training=training)
            gate_output = tf.expand_dims(gate_output, axis=-1)  # (batch_size, num_experts, 1)

            # 合并共享+特定专家输出：(batch_size, num_shared+num_specific, expert_units)
            all_expert_outputs = tf.concat([expert_outputs_shared, expert_outputs_specific[i]], axis=1)
            # 加权求和：(batch_size, expert_units)
            weighted_expert_output = all_expert_outputs * gate_output
            task_output = tf.reduce_sum(weighted_expert_output, axis=1)
            task_outputs.append(task_output)

        # ========== 4. 共享门控加权（非最后一层） ==========
        if not is_last_layer:
            # 共享门控输出：(batch_size, num_shared + num_tasks*num_specific)
            gate_output = cgc_config["shared_gate"](inputs[-1],training=training)
            gate_output = tf.expand_dims(gate_output, axis=-1)

            # 合并所有专家输出：共享 + 所有任务的特定专家
            all_expert_outputs = tf.concat([expert_outputs_shared] + expert_outputs_specific, axis=1)
            # 加权求和
            weighted_expert_output = all_expert_outputs * gate_output
            task_output = tf.reduce_sum(weighted_expert_output, axis=1)
            task_outputs.append(task_output)

        return task_outputs

    def call(self, inputs, training=None):
        """
        PLE模型前向传播
        :param inputs: 输入张量 (batch_size, input_dim)
        :param training: 是否训练模式（影响dropout/BN）
        :return: num_tasks 个任务特征列表，每个元素形状 (batch_size, expert_units)
        """
        # 初始化PLE输入：[任务1输入, 任务2输入, ..., 共享输入]
        ple_inputs = [inputs] * (self.num_tasks + 1)

        # 逐层执行CGC层
        for cgc_config in self.cgc_layers:
            ple_outputs = self.cgc_forward(ple_inputs, cgc_config,training=training)
            # 非最后一层，更新输入为当前层输出
            if not cgc_config["is_last_layer"]:
                ple_inputs = ple_outputs

        # 最终输出：[任务1特征, 任务2特征, ..., 任务N特征]
        return ple_outputs[:self.num_tasks]


class adatt(tf.keras.layers.Layer):
    def __init__(self, num_experts, num_tasks, expert_dim, task_dim, **kwargs):
        super(adatt, self).__init__(**kwargs)

