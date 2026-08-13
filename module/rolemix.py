import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight('scale', shape=[self.dim], initializer='ones')

    def call(self, x):
        rms = tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        return x * tf.cast(rms, x.dtype) * tf.cast(self.scale, x.dtype)


class SinkhornTokenMixing(tf.keras.layers.Layer):
    def __init__(self, token_count, iterations=4, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.token_count = token_count
        self.iterations = iterations
        self.temperature = temperature

    def build(self, input_shape):
        self.logits = self.add_weight('token_logits', shape=[self.token_count, self.token_count],
                                      initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.identity_bias = self.add_weight('identity_bias', shape=[], initializer='ones')

    def call(self, x):
        logits = 0.5 * (self.logits + tf.transpose(self.logits))
        logits += self.identity_bias * tf.eye(self.token_count, dtype=logits.dtype)
        matrix = tf.exp(logits / self.temperature)
        for _ in range(self.iterations):
            matrix /= tf.reduce_sum(matrix, axis=1, keepdims=True) + 1e-8
            matrix /= tf.reduce_sum(matrix, axis=0, keepdims=True) + 1e-8
        return tf.einsum('ij,bjd->bid', matrix, x)


class PerTokenSwiGLU(tf.keras.layers.Layer):
    def __init__(self, token_count, dim, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.token_count = token_count
        self.dim = dim
        self.hidden_dim = dim * hidden_ratio

    def build(self, input_shape):
        init = tf.keras.initializers.GlorotUniform()
        self.w_gate = self.add_weight('w_gate', [self.token_count, self.dim, self.hidden_dim], initializer=init)
        self.b_gate = self.add_weight('b_gate', [self.token_count, self.hidden_dim], initializer='zeros')
        self.w_up = self.add_weight('w_up', [self.token_count, self.dim, self.hidden_dim], initializer=init)
        self.b_up = self.add_weight('b_up', [self.token_count, self.hidden_dim], initializer='zeros')
        self.w_down = self.add_weight('w_down', [self.token_count, self.hidden_dim, self.dim], initializer=init)
        self.b_down = self.add_weight('b_down', [self.token_count, self.dim], initializer='zeros')

    def call(self, x):
        gate = tf.nn.swish(tf.einsum('btd,tdh->bth', x, self.w_gate) + self.b_gate)
        up = tf.einsum('btd,tdh->bth', x, self.w_up) + self.b_up
        return tf.einsum('bth,thd->btd', gate * up, self.w_down) + self.b_down


class UniMixingLiteBlock(tf.keras.layers.Layer):
    def __init__(self, token_count, dim, hidden_ratio=2, sinkhorn_iters=4, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.mix_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.mix = SinkhornTokenMixing(token_count, sinkhorn_iters, temperature)
        self.ffn = PerTokenSwiGLU(token_count, dim, hidden_ratio)

    def build(self, input_shape):
        self.alpha = self.add_weight('alpha', shape=[], initializer=tf.keras.initializers.Constant(0.1))
        self.beta = self.add_weight('beta', shape=[], initializer=tf.keras.initializers.Constant(0.1))

    def call(self, x):
        x = x + self.alpha * self.mix(self.mix_norm(x))
        return x + self.beta * self.ffn(self.ffn_norm(x))


class RoleMix(tf.keras.layers.Layer):
    def __init__(self, semantic_tokens=16, sequence_tokens=6, dim=256, num_blocks=2,
                 hidden_ratio=2, sinkhorn_iters=4, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.semantic_tokens = semantic_tokens
        self.sequence_tokens = sequence_tokens
        self.token_count = 1 + semantic_tokens + sequence_tokens
        self.dim = dim
        self.blocks = [UniMixingLiteBlock(self.token_count, dim, hidden_ratio, sinkhorn_iters, temperature,
                                          name='unimixing_lite_{}'.format(i)) for i in range(num_blocks)]

    def build(self, input_shape):
        self.global_token = self.add_weight('global_token', [1, 1, self.dim],
                                            initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, inputs):
        semantic_tokens, sequence_tokens = inputs
        batch = tf.shape(semantic_tokens)[0]
        global_token = tf.tile(self.global_token, [batch, 1, 1])
        x = tf.concat([global_token, sequence_tokens, semantic_tokens], axis=1)
        for block in self.blocks:
            x = block(x)
        global_out = x[:, 0, :]
        sequence_out = tf.reduce_mean(x[:, 1:1 + self.sequence_tokens, :], axis=1)
        semantic_out = tf.reshape(
            x[:, 1 + self.sequence_tokens:, :],
            [batch, self.semantic_tokens * self.dim])
        return tf.concat([global_out, sequence_out, semantic_out], axis=-1)
