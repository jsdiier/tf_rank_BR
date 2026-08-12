import tensorflow as tf

from .activation import GELU
from logger import logger


class MultiHeadTokenMixing(tf.keras.layers.Layer):
    """Parameter-free RankMixer token mixing with H equal to T."""

    def __init__(self, token_count, token_dim, **kwargs):
        super().__init__(**kwargs)
        if token_dim % token_count != 0:
            raise ValueError('token_dim must be divisible by token_count')
        self.token_count = token_count
        self.token_dim = token_dim
        self.head_dim = token_dim // token_count

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.token_count, self.token_count, self.head_dim])
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [batch_size, self.token_count, self.token_dim])


class BatchedPerTokenFFN(tf.keras.layers.Layer):
    """Independent FFN weights per token, executed as batched einsums."""

    def __init__(self, token_count, token_dim, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.token_count = token_count
        self.token_dim = token_dim
        self.hidden_dim = hidden_ratio * token_dim
        self.gelu = GELU()

    def build(self, input_shape):
        glorot = tf.keras.initializers.GlorotUniform()
        self.kernel_up = self.add_weight(
            name='kernel_up', shape=[self.token_count, self.token_dim, self.hidden_dim], initializer=glorot)
        self.bias_up = self.add_weight(
            name='bias_up', shape=[self.token_count, self.hidden_dim], initializer='zeros')
        self.kernel_down = self.add_weight(
            name='kernel_down', shape=[self.token_count, self.hidden_dim, self.token_dim], initializer=glorot)
        self.bias_down = self.add_weight(
            name='bias_down', shape=[self.token_count, self.token_dim], initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        hidden = tf.einsum('btd,tdh->bth', x, self.kernel_up) + self.bias_up
        hidden = self.gelu(hidden)
        return tf.einsum('bth,thd->btd', hidden, self.kernel_down) + self.bias_down


class RankMixerBlock(tf.keras.layers.Layer):
    def __init__(self, token_count, token_dim, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.token_mixing = MultiHeadTokenMixing(token_count, token_dim)
        self.token_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.per_token_ffn = BatchedPerTokenFFN(token_count, token_dim, hidden_ratio)
        self.ffn_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def call(self, x, training=None):
        x = self.token_ln(x + self.token_mixing(x))
        return self.ffn_ln(x + self.per_token_ffn(x))


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
            'Paper RankMixer initialized: tokens={}, token_dim={}, blocks={}, heads={}, '
            'per_token_ffn={}->{}->{}'.format(
                t, token_dim, num_blocks, t, token_dim, hidden_ratio * token_dim, token_dim))

    def call(self, x):
        if x.shape.rank != 3 or x.shape[1] != self.t or x.shape[2] != self.token_dim:
            raise ValueError(
                'RankMixer expects [B, {}, {}], got {}'.format(self.t, self.token_dim, x.shape))
        for block in self.blocks:
            x = block(x, training=self.training)
        return tf.reduce_mean(x, axis=1)
