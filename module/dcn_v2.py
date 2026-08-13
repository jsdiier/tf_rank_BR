import tensorflow as tf
from tensorflow.keras import regularizers


class CrossNetMixLayer(tf.keras.layers.Layer):
    """One low-rank mixture-of-experts cross layer from DCN-V2."""

    def __init__(self, input_dim, low_rank=32, num_experts=4, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.low_rank = low_rank
        self.num_experts = num_experts
        reg = regularizers.l2(l2_reg)

        self.gates = [
            tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=reg,
                                  name='gate_{}'.format(index))
            for index in range(num_experts)
        ]
        self.v_layers = [
            tf.keras.layers.Dense(low_rank, use_bias=False, kernel_regularizer=reg,
                                  name='v_{}'.format(index))
            for index in range(num_experts)
        ]
        self.c_layers = [
            tf.keras.layers.Dense(low_rank, use_bias=False, kernel_regularizer=reg,
                                  name='c_{}'.format(index))
            for index in range(num_experts)
        ]
        self.u_layers = [
            tf.keras.layers.Dense(input_dim, use_bias=True, kernel_regularizer=reg,
                                  name='u_{}'.format(index))
            for index in range(num_experts)
        ]

    def call(self, inputs):
        x0, xl = inputs
        expert_outputs = []
        gate_logits = []
        for gate, v_layer, c_layer, u_layer in zip(
                self.gates, self.v_layers, self.c_layers, self.u_layers):
            low_rank = tf.nn.tanh(v_layer(xl))
            low_rank = tf.nn.tanh(c_layer(low_rank))
            cross = x0 * u_layer(low_rank)
            expert_outputs.append(cross)
            gate_logits.append(gate(xl))

        experts = tf.stack(expert_outputs, axis=1)
        weights = tf.nn.softmax(tf.concat(gate_logits, axis=1), axis=1)
        mixed_cross = tf.reduce_sum(experts * weights[:, :, None], axis=1)
        return xl + mixed_cross


class DCNV2(tf.keras.layers.Layer):
    """Stacked low-rank CrossNet-Mix with shape-preserving residuals."""

    def __init__(self, input_dim, num_layers=2, low_rank=32,
                 num_experts=4, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.cross_layers = [
            CrossNetMixLayer(
                input_dim=input_dim,
                low_rank=low_rank,
                num_experts=num_experts,
                l2_reg=l2_reg,
                name='cross_mix_{}'.format(index))
            for index in range(num_layers)
        ]

    def call(self, inputs):
        x0 = inputs
        xl = inputs
        for cross_layer in self.cross_layers:
            xl = cross_layer([x0, xl])
        return xl
