import tensorflow as tf

from .activation import GELU
from logger import logger


class TokenMixingMLP(tf.keras.layers.Layer):
    """Learn interactions along the token axis for every hidden channel."""

    def __init__(self, token_count, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = hidden_ratio * token_count
        self.dense_up = tf.keras.layers.Dense(hidden_dim, activation=GELU())
        self.dense_down = tf.keras.layers.Dense(token_count)

    def call(self, x):
        x = tf.transpose(x, [0, 2, 1])
        x = self.dense_down(self.dense_up(x))
        return tf.transpose(x, [0, 2, 1])


class SwiGLU(tf.keras.layers.Layer):
    """One shared batched FFN for all tokens."""

    def __init__(self, token_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.gate_dense = tf.keras.layers.Dense(hidden_dim)
        self.up_dense = tf.keras.layers.Dense(hidden_dim)
        self.down_dense = tf.keras.layers.Dense(token_dim)

    def call(self, x):
        gate = tf.keras.activations.swish(self.gate_dense(x))
        return self.down_dense(gate * self.up_dense(x))


class RankMixerBlock(tf.keras.layers.Layer):
    def __init__(self, token_count, token_dim, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.token_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.token_mixing = TokenMixingMLP(token_count, hidden_ratio=2)
        self.ffn_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.shared_swiglu = SwiGLU(token_dim, hidden_ratio * token_dim)

    def call(self, x, training=None):
        x = x + self.token_mixing(self.token_ln(x))
        return x + self.shared_swiglu(self.ffn_ln(x))


class RankMixer(tf.keras.layers.Layer):
    def __init__(self, t, token_dim, num_blocks=2, num_heads=None, num_experts=1,
                 hidden_ratio=2, l1_coeff=0.0, training=False, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.token_dim = token_dim
        self.training = training
        self.blocks = [
            RankMixerBlock(t, token_dim, hidden_ratio, name='rankmixer_block_{}'.format(index))
            for index in range(num_blocks)
        ]
        logger.info(
            'RankMixer initialized: tokens={}, token_dim={}, blocks={}, '
            'token_mlp={}->{}->{}, shared_ffn={}->{}->{}'.format(
                t, token_dim, num_blocks, t, 2 * t, t,
                token_dim, hidden_ratio * token_dim, token_dim))

    def call(self, x):
        if x.shape.rank != 3 or x.shape[1] != self.t or x.shape[2] != self.token_dim:
            raise ValueError(
                'RankMixer expects [B, {}, {}], got {}'.format(self.t, self.token_dim, x.shape))
        for block in self.blocks:
            x = block(x, training=self.training)
        return tf.reduce_mean(x, axis=1)
